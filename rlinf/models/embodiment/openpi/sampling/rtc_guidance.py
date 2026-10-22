# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RTC overlap guidance for the vendored Pi0 eval sampler.

``RTCGuidanceContext`` and ``build_rtc_target_and_mask`` are the shared
protocol used by both ``openpi`` and the leftover OpenPI PyTorch
sampler; the Euler loop below is the vendored-Pi0 implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from rlinf.models.embodiment.openpi.modules.model import preprocess_observation
from rlinf.models.embodiment.openpi.sampling import rl_sampler


@dataclass
class RTCGuidanceContext:
    """RTC context passed from rollout runtime to the OpenPI sampler."""

    prev_model_actions: torch.Tensor | None = None  # [B, H, A_model]
    executed_horizon: int = 0
    delay_steps: int = 0
    guided_action_indices: tuple[int, ...] | None = None

    def get_prev_remaining(self) -> torch.Tensor | None:
        if self.prev_model_actions is None:
            return None
        executed_horizon = int(max(self.executed_horizon, 0))
        if executed_horizon >= self.prev_model_actions.shape[1]:
            return None
        return self.prev_model_actions[:, executed_horizon:, :]


def _rtc_guidance_weight_at_time(
    time_value: torch.Tensor, max_guidance_weight: float
) -> float:
    """Return the guidance weight for one denoising step.

    The schedule is U-shaped: it is capped at ``max_guidance_weight`` near
    both ends of the trajectory and reaches its minimum at the midpoint.
    """
    time_float = float(time_value.detach().item())
    tau = 1.0 - time_float
    if tau <= 0.0:
        return float(max_guidance_weight)
    if time_float <= 0.0:
        return 0.0
    inverse_r_squared = (time_float**2 + tau**2) / (time_float**2)
    guidance = (time_float / tau) * inverse_r_squared
    return min(float(max_guidance_weight), float(guidance))


def build_rtc_target_and_mask(
    prev_remaining: torch.Tensor | None,
    horizon: int,
    action_dim: int,
    delay_steps: int,
    device: torch.device,
    dtype: torch.dtype,
    guided_action_indices: tuple[int, ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the overlap target and a soft per-dimension guidance mask.

    The delay prefix is stale by the time the new chunk returns, so it is left
    unguided. The still-relevant overlap decays with a squared cosine from one
    to zero. ``guided_action_indices`` restricts guidance to selected action
    dimensions; out-of-range indices are ignored.
    """
    batch_size = 1 if prev_remaining is None else prev_remaining.shape[0]
    target = torch.zeros((batch_size, horizon, action_dim), device=device, dtype=dtype)
    mask = torch.zeros((batch_size, horizon, action_dim), device=device, dtype=dtype)

    if prev_remaining is None or prev_remaining.numel() == 0:
        return target, mask

    overlap = min(prev_remaining.shape[1], horizon)
    if overlap <= 0:
        return target, mask

    target[:, :overlap] = prev_remaining[:, :overlap].to(device=device, dtype=dtype)
    soft_start = min(max(int(delay_steps), 0), overlap)
    if soft_start >= overlap:
        return target, mask

    soft_length = overlap - soft_start
    ramp = torch.linspace(0.0, 1.0, soft_length, device=device, dtype=dtype)
    soft = torch.cos(0.5 * math.pi * ramp).square()

    dimension_mask = torch.zeros((action_dim,), device=device, dtype=dtype)
    if guided_action_indices is None:
        dimension_mask.fill_(1.0)
    else:
        valid_indices = [
            int(index)
            for index in guided_action_indices
            if 0 <= int(index) < action_dim
        ]
        if valid_indices:
            dimension_mask[valid_indices] = 1.0

    mask[:, soft_start:overlap, :] = soft[None, :, None] * dimension_mask[None, None, :]
    return target, mask


@torch.no_grad()
def sample_actions_with_rtc_guidance(
    pi0_model,
    observation,
    rtc_context: RTCGuidanceContext,
    *,
    num_steps: int,
    noise: torch.Tensor | None = None,
    rng: torch.Generator | None = None,
    guidance_clip: float = 5.0,
) -> torch.Tensor:
    """Euler ODE sampling with overlap guidance against the previous chunk."""
    observation = preprocess_observation(observation, train=False)
    B = observation.state.shape[0]
    device = observation.state.device
    if noise is None:
        noise = torch.randn(
            B,
            pi0_model.action_horizon,
            pi0_model.action_dim,
            device=device,
            dtype=torch.float32,
            generator=rng,
        )
    else:
        noise = noise.to(device=device, dtype=torch.float32)

    prefix_out, prefix_mask, kv_cache = pi0_model.build_prefix_cache(observation)
    del prefix_out

    prev_remaining = (
        rtc_context.get_prev_remaining() if rtc_context is not None else None
    )
    target, mask = build_rtc_target_and_mask(
        prev_remaining=prev_remaining,
        horizon=pi0_model.action_horizon,
        action_dim=pi0_model.action_dim,
        delay_steps=0 if rtc_context is None else rtc_context.delay_steps,
        device=device,
        dtype=noise.dtype,
        guided_action_indices=(
            None if rtc_context is None else rtc_context.guided_action_indices
        ),
    )

    x_t = noise
    paper_tau = torch.tensor(0.0, device=device, dtype=noise.dtype)
    dt = torch.tensor(1.0 / num_steps, device=device, dtype=noise.dtype)
    idx_step = torch.empty(B, device=device, dtype=torch.long)

    for idx in range(num_steps):
        t_val = float(rl_sampler.get_timesteps(num_steps, device)[idx].item())
        t_tensor = torch.full((B,), t_val, device=device, dtype=torch.float32)
        suffix_act = pi0_model.run_suffix(
            observation, x_t, t_tensor, kv_cache, prefix_mask
        )
        v_t = pi0_model.velocity_from_suffix(suffix_act).to(torch.float32)
        idx_step.fill_(idx)
        x_t_mean, x_t_std = rl_sampler.sample_mean_var(
            x_t.to(torch.float32),
            v_t,
            idx_step,
            noise_method="flow_ode",
            noise_level=0.0,
            num_steps=num_steps,
        )
        if prev_remaining is not None and prev_remaining.numel() > 0:
            guidance_term = (target - x_t_mean) * mask
            guidance_scale = torch.clamp(
                (1.0 - paper_tau) / torch.clamp(paper_tau + 1e-4, min=1e-4),
                max=float(guidance_clip),
            )
            x_t_mean = x_t_mean + guidance_scale * guidance_term
        step_noise = torch.randn(
            x_t.shape, device=device, dtype=torch.float32, generator=rng
        )
        x_t = x_t_mean + step_noise * x_t_std
        paper_tau = paper_tau + dt
    return x_t


@torch.no_grad()
def sample_actions_with_vjp_rtc_guidance(
    pi0_model,
    observation,
    rtc_context: RTCGuidanceContext,
    *,
    num_steps: int,
    noise: torch.Tensor | None = None,
    rng: torch.Generator | None = None,
    guidance_weight: float = 5.0,
) -> torch.Tensor:
    """Euler ODE sampling with VJP guidance on the denoised action estimate.

    The guidance weight follows the NERO RTC schedule instead of staying
    constant across denoising steps.
    """
    observation = preprocess_observation(observation, train=False)
    batch_size = observation.state.shape[0]
    device = observation.state.device
    if noise is None:
        noise = torch.randn(
            batch_size,
            pi0_model.action_horizon,
            pi0_model.action_dim,
            device=device,
            dtype=torch.float32,
            generator=rng,
        )
    else:
        noise = noise.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        _, prefix_mask, kv_cache = pi0_model.build_prefix_cache(observation)

    prev_remaining = rtc_context.get_prev_remaining()
    target, mask = build_rtc_target_and_mask(
        prev_remaining=prev_remaining,
        horizon=pi0_model.action_horizon,
        action_dim=pi0_model.action_dim,
        delay_steps=rtc_context.delay_steps,
        device=device,
        dtype=noise.dtype,
        guided_action_indices=rtc_context.guided_action_indices,
    )
    has_guidance = bool(torch.count_nonzero(mask).item())
    max_guidance_weight = float(guidance_weight)
    runtime_guidance_weights: list[float] = []

    x_t = noise
    timesteps = rl_sampler.get_timesteps(num_steps, device)
    for index in range(num_steps):
        t = timesteps[index]
        t_batch = t.expand(batch_size)
        dt = timesteps[index + 1] - t
        step_guidance_weight = _rtc_guidance_weight_at_time(t, max_guidance_weight)
        runtime_guidance_weights.append(step_guidance_weight)

        with torch.enable_grad():
            x_var = x_t.detach().requires_grad_(has_guidance)
            suffix_output = pi0_model.run_suffix(
                observation,
                x_var,
                t_batch,
                kv_cache,
                prefix_mask,
            )
            velocity = pi0_model.velocity_from_suffix(suffix_output).to(torch.float32)

            if has_guidance:
                denoised = x_var - t_batch[:, None, None] * velocity
                weighted_error = (target - denoised) * mask
                correction = torch.autograd.grad(
                    outputs=denoised,
                    inputs=x_var,
                    grad_outputs=weighted_error,
                    retain_graph=False,
                    create_graph=False,
                )[0]
                velocity = velocity - step_guidance_weight * correction

        x_t = (x_var + dt * velocity).detach()

    pi0_model._last_rtc_guidance_weights = runtime_guidance_weights
    return x_t
