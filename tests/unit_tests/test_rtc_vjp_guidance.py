# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

import pytest
import torch

from rlinf.models.embodiment.openpi.modules.model import Observation
from rlinf.models.embodiment.openpi.sampling.rtc_guidance import (
    RTCGuidanceContext,
    _rtc_guidance_weight_at_time,
    build_rtc_target_and_mask,
    sample_actions_with_rtc_guidance,
    sample_actions_with_vjp_rtc_guidance,
)


def _musa_available() -> bool:
    return hasattr(torch, "musa") and torch.musa.is_available()


def test_rtc_mask_skips_delay_prefix_and_selects_action_dimensions():
    previous = torch.ones(1, 6, 4)

    target, mask = build_rtc_target_and_mask(
        previous,
        horizon=6,
        action_dim=4,
        delay_steps=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        guided_action_indices=(0, 2),
    )

    assert torch.equal(target, previous)
    assert mask.shape == (1, 6, 4)
    assert torch.count_nonzero(mask[:, :2]).item() == 0
    assert torch.count_nonzero(mask[..., 1]).item() == 0
    assert torch.count_nonzero(mask[..., 3]).item() == 0
    assert mask[0, 2, 0].item() == pytest.approx(1.0)
    assert mask[0, -1, 0].item() == pytest.approx(0.0, abs=1e-6)


def test_rtc_mask_ignores_out_of_range_action_dimensions():
    _, mask = build_rtc_target_and_mask(
        torch.ones(1, 2, 3),
        horizon=2,
        action_dim=3,
        delay_steps=0,
        device=torch.device("cpu"),
        dtype=torch.float32,
        guided_action_indices=(-1, 3),
    )
    assert torch.count_nonzero(mask).item() == 0

    _, mask = build_rtc_target_and_mask(
        torch.ones(1, 2, 3),
        horizon=2,
        action_dim=3,
        delay_steps=0,
        device=torch.device("cpu"),
        dtype=torch.float32,
        guided_action_indices=(-1, 1, 3),
    )
    assert torch.count_nonzero(mask[..., 0]).item() == 0
    assert torch.count_nonzero(mask[..., 1]).item() > 0
    assert torch.count_nonzero(mask[..., 2]).item() == 0


def test_rtc_guidance_weight_at_time_schedule():
    timesteps = torch.linspace(1.0, 0.1, 10)
    weights = [_rtc_guidance_weight_at_time(t, 5.0) for t in timesteps]
    assert weights == pytest.approx(
        [5.0, 5.0, 4.25, 2.762, 2.167, 2.0, 2.167, 2.762, 4.25, 5.0],
        abs=1e-3,
    )


class _LinearVelocityModel:
    action_horizon = 4
    action_dim = 3

    def build_prefix_cache(self, observation):
        return None, None, None

    def run_suffix(self, observation, x_t, t_tensor, kv_cache, prefix_mask):
        return x_t

    def velocity_from_suffix(self, suffix_output):
        return suffix_output


def _observation(device: torch.device) -> Observation:
    images = {
        key: torch.zeros(1, 224, 224, 3, device=device)
        for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    }
    image_masks = {
        key: torch.ones(1, dtype=torch.bool, device=device) for key in images
    }
    return Observation(
        images=images,
        image_masks=image_masks,
        state=torch.zeros(1, 8, device=device),
    )


@pytest.mark.parametrize("device_name", ["cpu", "musa"])
def test_vjp_guidance_changes_guided_sampling_result(device_name: str):
    if device_name == "musa" and not _musa_available():
        pytest.skip("MUSA is unavailable")

    device = torch.device(device_name)
    model = _LinearVelocityModel()
    observation = _observation(device)
    noise = torch.full((1, model.action_horizon, model.action_dim), 0.5, device=device)
    previous = torch.ones_like(noise)

    unguided = sample_actions_with_vjp_rtc_guidance(
        model,
        observation,
        RTCGuidanceContext(),
        num_steps=4,
        noise=noise,
    )
    guided = sample_actions_with_vjp_rtc_guidance(
        model,
        observation,
        RTCGuidanceContext(
            prev_model_actions=previous,
            executed_horizon=0,
            delay_steps=0,
            guided_action_indices=(0, 2),
        ),
        num_steps=4,
        noise=noise,
        guidance_weight=2.0,
    )

    assert torch.isfinite(guided).all()
    assert not torch.allclose(guided[:, :, 0], unguided[:, :, 0])
    assert torch.allclose(guided[:, :, 1], unguided[:, :, 1])
    assert not torch.allclose(guided[:, :, 2], unguided[:, :, 2])


@pytest.mark.parametrize("device_name", ["cpu", "musa"])
def test_vjp_guidance_leaves_delay_prefix_unguided(device_name: str):
    if device_name == "musa" and not _musa_available():
        pytest.skip("MUSA is unavailable")

    device = torch.device(device_name)
    model = _LinearVelocityModel()
    observation = _observation(device)
    noise = torch.full((1, model.action_horizon, model.action_dim), 0.5, device=device)

    unguided = sample_actions_with_vjp_rtc_guidance(
        model,
        observation,
        RTCGuidanceContext(),
        num_steps=4,
        noise=noise,
    )
    guided = sample_actions_with_vjp_rtc_guidance(
        model,
        observation,
        RTCGuidanceContext(
            prev_model_actions=torch.zeros_like(noise),
            executed_horizon=0,
            delay_steps=1,
            guided_action_indices=(0,),
        ),
        num_steps=4,
        noise=noise,
        guidance_weight=5.0,
    )

    assert torch.allclose(guided[:, 0, :], unguided[:, 0, :])
    assert not torch.allclose(guided[:, 1:, 0], unguided[:, 1:, 0])
    assert model._last_rtc_guidance_weights == pytest.approx(
        [5.0, 3.333, 2.0, 3.333], abs=1e-3
    )


@pytest.mark.parametrize("device_name", ["cpu", "musa"])
def test_approx_guidance_skips_delay_prefix_and_unguided_dimensions(
    device_name: str,
):
    if device_name == "musa" and not _musa_available():
        pytest.skip("MUSA is unavailable")

    device = torch.device(device_name)
    model = _LinearVelocityModel()
    observation = _observation(device)
    noise = torch.full((1, model.action_horizon, model.action_dim), 0.5, device=device)

    unguided = sample_actions_with_rtc_guidance(
        model,
        observation,
        RTCGuidanceContext(),
        num_steps=4,
        noise=noise,
    )
    guided = sample_actions_with_rtc_guidance(
        model,
        observation,
        RTCGuidanceContext(
            prev_model_actions=torch.zeros_like(noise),
            executed_horizon=0,
            delay_steps=1,
            guided_action_indices=(0, 2),
        ),
        num_steps=4,
        noise=noise,
    )

    assert torch.isfinite(guided).all()
    assert torch.allclose(guided[:, 0, :], unguided[:, 0, :])
    assert not torch.allclose(guided[:, 1:, 0], unguided[:, 1:, 0])
    assert torch.allclose(guided[:, :, 1], unguided[:, :, 1])
    assert not torch.allclose(guided[:, :, 2], unguided[:, :, 2])
