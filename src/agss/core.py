from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


Array = np.ndarray
EdgeProvider = Callable[[Array, float], Array]


@dataclass
class AgssRunArtifacts:
    output_image: Array
    edge_maps: dict[float, Array]
    snapshots: dict[int, Array]
    summary: dict[str, Any]


def parse_alpha_keys(value: str | list[float] | tuple[float, ...] | None) -> list[float]:
    if value is None:
        return [3.0, 2.0, 1.0, 0.0]
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    items = [item.strip() for item in text.split(",") if item.strip()]
    if not items:
        raise ValueError("alpha_keys is empty")
    return [float(item) for item in items]


def _select_device(device: str | None) -> torch.device:
    if device in (None, "", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _to_image_tensor(image_rgb: Array, device: torch.device) -> torch.Tensor:
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 image, got {image_rgb.shape}")
    image = np.clip(image_rgb.astype(np.float32, copy=False), 0.0, 1.0)
    return torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).to(device)


def _to_edge_tensor(edge_map: Array, device: torch.device) -> torch.Tensor:
    if edge_map.ndim != 2:
        raise ValueError(f"Expected HxW edge map, got {edge_map.shape}")
    return torch.from_numpy(np.clip(edge_map.astype(np.float32, copy=False), 0.0, 1.0)).unsqueeze(0).unsqueeze(0).to(device)


def _compute_denominator(weight_map: torch.Tensor, radius: int, xi: float) -> torch.Tensor:
    kernel_size = 2 * radius + 1
    base_kernel = torch.ones((1, 1, kernel_size, kernel_size), device=weight_map.device, dtype=weight_map.dtype)
    padded = F.pad(weight_map, (radius, radius, radius, radius), mode="reflect")
    denominator = F.conv2d(padded, base_kernel, padding=0) + float(xi)
    floor = max(float(xi), torch.finfo(weight_map.dtype).eps)
    return torch.clamp(denominator, min=floor)


def _weighted_average_step(
    image_t: torch.Tensor,
    weight_map: torch.Tensor,
    denominator: torch.Tensor,
    radius: int,
    xi: float,
) -> torch.Tensor:
    channels = image_t.shape[1]
    kernel_size = 2 * radius + 1
    base_kernel = torch.ones((1, 1, kernel_size, kernel_size), device=image_t.device, dtype=image_t.dtype)
    grouped_kernel = base_kernel.repeat(channels, 1, 1, 1)
    padded_image = F.pad(image_t, (radius, radius, radius, radius), mode="reflect")
    padded_weight = F.pad(weight_map, (radius, radius, radius, radius), mode="reflect")
    weighted_input = padded_image * padded_weight
    weighted_sum = F.conv2d(weighted_input, grouped_kernel, padding=0, groups=channels)
    stabilized_sum = weighted_sum + float(xi) * image_t
    return stabilized_sum / denominator


def run_agss(
    image_rgb: Array,
    edge_provider: EdgeProvider,
    *,
    alpha_keys: str | list[float] | tuple[float, ...] | None = None,
    n_iters: int = 300,
    radius: int = 1,
    delta_base: float = 1e-6,
    decay: float = 0.4,
    xi: float = 1e-6,
    device: str | None = "auto",
    snapshot_iterations: list[int] | tuple[int, ...] | None = None,
    provider_metadata: dict[str, Any] | None = None,
) -> AgssRunArtifacts:
    alpha_values = parse_alpha_keys(alpha_keys)
    if n_iters < 0:
        raise ValueError("n_iters must be non-negative")
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if xi < 0:
        raise ValueError("xi must be non-negative")

    snapshot_iteration_set = {
        int(iteration)
        for iteration in (snapshot_iterations or [])
        if int(iteration) > 0
    }
    if any(iteration > int(n_iters) for iteration in snapshot_iteration_set):
        raise ValueError("snapshot_iterations must be <= n_iters")

    torch_device = _select_device(device)
    total_start = time.time()
    edge_start = time.time()
    edge_maps_np: dict[float, Array] = {}
    stage_weights: dict[float, torch.Tensor] = {}
    stage_denominators: dict[float, torch.Tensor] = {}

    for alpha in alpha_values:
        edge_map = edge_provider(image_rgb, float(alpha))
        if edge_map.shape != image_rgb.shape[:2]:
            raise ValueError(
                f"Edge map shape mismatch for alpha={alpha}: {edge_map.shape} vs {image_rgb.shape[:2]}"
            )
        alpha_key = float(alpha)
        edge_maps_np[alpha_key] = np.clip(edge_map.astype(np.float32, copy=False), 0.0, 1.0)
        stage_boundary = _to_edge_tensor(edge_maps_np[alpha_key], torch_device)
        stage_weight = torch.clamp(1.0 - stage_boundary, 0.0, 1.0)
        stage_weights[alpha_key] = stage_weight
        stage_denominators[alpha_key] = _compute_denominator(stage_weight, radius, xi)

    edge_time_sec = time.time() - edge_start

    image_t = _to_image_tensor(image_rgb, torch_device)
    current_stage = 0
    current_alpha = float(alpha_values[current_stage])
    current_weight = stage_weights[current_alpha]
    current_denominator = stage_denominators[current_alpha]
    delta_target = float(delta_base)
    mad_prev = float("inf")
    stage_transitions: list[dict[str, Any]] = []
    iteration_log: list[dict[str, Any]] = []
    snapshots: dict[int, Array] = {}
    iter_start = time.time()

    for iteration_index in range(int(n_iters)):
        image_prev = image_t
        next_image = _weighted_average_step(image_prev, current_weight, current_denominator, radius, xi)
        mad_curr = torch.mean(torch.abs(next_image - image_prev)).item()
        delta_mad = abs(mad_curr - mad_prev) if np.isfinite(mad_prev) else None
        converged = delta_mad is not None and delta_mad < delta_target
        image_t = next_image

        iteration_log.append(
            {
                "iteration": iteration_index + 1,
                "stage_index": current_stage,
                "alpha": current_alpha,
                "mad_curr": float(mad_curr),
                "mad_prev": None if not np.isfinite(mad_prev) else float(mad_prev),
                "delta_mad": None if delta_mad is None else float(delta_mad),
                "delta_target": float(delta_target),
                "converged": bool(converged),
            }
        )

        completed_iteration = iteration_index + 1
        if completed_iteration in snapshot_iteration_set:
            snapshots[completed_iteration] = np.clip(
                image_t.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0),
                0.0,
                1.0,
            )

        last_stage = current_stage >= len(alpha_values) - 1
        if converged and not last_stage:
            previous_alpha = current_alpha
            current_stage += 1
            current_alpha = float(alpha_values[current_stage])
            current_weight = stage_weights[current_alpha]
            current_denominator = stage_denominators[current_alpha]
            delta_target = float(delta_base) * (float(decay) ** current_stage)
            mad_prev = float("inf")
            stage_transitions.append(
                {
                    "after_iteration": iteration_index + 1,
                    "from_stage_index": current_stage - 1,
                    "to_stage_index": current_stage,
                    "from_alpha": previous_alpha,
                    "to_alpha": current_alpha,
                }
            )
        else:
            mad_prev = mad_curr

    smoothing_time_sec = time.time() - iter_start
    output_image = image_t.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    output_image = np.clip(output_image, 0.0, 1.0)

    summary = {
        "alpha_keys": [float(v) for v in alpha_values],
        "n_iters": int(n_iters),
        "radius": int(radius),
        "delta_base": float(delta_base),
        "decay": float(decay),
        "xi": float(xi),
        "device": str(torch_device),
        "snapshot_iterations": sorted(snapshot_iteration_set),
        "timings": {
            "edge_map_generation_sec": edge_time_sec,
            "smoothing_sec": smoothing_time_sec,
            "total_sec": time.time() - total_start,
        },
        "stage_transitions": stage_transitions,
        "final_stage_index": current_stage,
        "final_alpha": current_alpha,
        "iterations": iteration_log,
    }
    if provider_metadata:
        summary["provider"] = provider_metadata
    return AgssRunArtifacts(
        output_image=output_image,
        edge_maps=edge_maps_np,
        snapshots=snapshots,
        summary=summary,
    )
