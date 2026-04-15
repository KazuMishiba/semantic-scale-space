from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core import AgssRunArtifacts, EdgeProvider, run_agss
from ..metrics.rhi import GaussianReferenceRHI


Array = np.ndarray


@dataclass
class E1PartialArtifacts:
    output_image: Array
    target_level: str
    target_sigma: float
    target_rhi: float
    selected_iteration: int
    selected_rhi: float
    agss_artifacts: AgssRunArtifacts
    summary: dict[str, Any]


def build_candidate_iterations(n_iters: int, stride: int = 5) -> list[int]:
    if n_iters < 0:
        raise ValueError("n_iters must be non-negative")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if n_iters == 0:
        return [0]

    iterations = {0, 1, int(n_iters)}
    iterations.update(range(stride, int(n_iters) + 1, stride))
    return sorted(iterations)


def _normalize_candidate_iterations(candidate_iterations: list[int] | tuple[int, ...], n_iters: int) -> list[int]:
    normalized = sorted({int(iteration) for iteration in candidate_iterations})
    if not normalized:
        raise ValueError("candidate_iterations must not be empty")
    if normalized[0] < 0:
        raise ValueError("candidate_iterations must be non-negative")
    if normalized[-1] > int(n_iters):
        raise ValueError("candidate_iterations must be <= n_iters")
    if 0 not in normalized:
        normalized.insert(0, 0)
    return normalized


def _build_candidate_record(
    iteration: int,
    candidate_image: Array,
    rhi_calculator: GaussianReferenceRHI,
    target_rhi: float,
    iteration_state: dict[str, Any] | None,
) -> dict[str, Any]:
    achieved_rhi = float(rhi_calculator.compute_rhi(candidate_image))
    abs_error = abs(achieved_rhi - target_rhi)
    record = {
        "iteration": int(iteration),
        "achieved_rhi": achieved_rhi,
        "target_rhi": float(target_rhi),
        "absolute_error": float(abs_error),
    }
    if iteration_state is not None:
        record.update(
            {
                "stage_index": int(iteration_state["stage_index"]),
                "alpha": float(iteration_state["alpha"]),
                "mad_curr": float(iteration_state["mad_curr"]),
                "delta_target": float(iteration_state["delta_target"]),
                "converged": bool(iteration_state["converged"]),
            }
        )
    else:
        record.update(
            {
                "stage_index": None,
                "alpha": None,
                "mad_curr": None,
                "delta_target": None,
                "converged": None,
            }
        )
    return record


def run_e1_partial_single(
    image_rgb: Array,
    gt_boundary: Array,
    edge_provider: EdgeProvider,
    *,
    target_level: str = "sigma2",
    candidate_iterations: list[int] | tuple[int, ...] | None = None,
    candidate_stride: int = 5,
    alpha_keys: str | list[float] | tuple[float, ...] | None = None,
    n_iters: int = 300,
    radius: int = 1,
    delta_base: float = 1e-6,
    decay: float = 0.4,
    xi: float = 1e-6,
    device: str | None = "auto",
    dilation_radius: int = 6,
    window_size: int = 7,
    provider_metadata: dict[str, Any] | None = None,
) -> E1PartialArtifacts:
    if gt_boundary.shape != image_rgb.shape[:2]:
        raise ValueError(
            f"GT boundary shape mismatch: expected {image_rgb.shape[:2]}, got {gt_boundary.shape}"
        )

    rhi_calculator = GaussianReferenceRHI(
        image_rgb,
        gt_boundary,
        dilation_radius=dilation_radius,
        window_size=window_size,
    )
    target = rhi_calculator.get_target(target_level)
    selected_candidate_iterations = (
        _normalize_candidate_iterations(candidate_iterations, n_iters)
        if candidate_iterations is not None
        else build_candidate_iterations(n_iters, candidate_stride)
    )
    snapshot_iterations = [iteration for iteration in selected_candidate_iterations if iteration > 0]

    agss_artifacts = run_agss(
        image_rgb,
        edge_provider,
        alpha_keys=alpha_keys,
        n_iters=n_iters,
        radius=radius,
        delta_base=delta_base,
        decay=decay,
        xi=xi,
        device=device,
        snapshot_iterations=snapshot_iterations,
        provider_metadata=provider_metadata,
    )

    candidate_images: dict[int, Array] = {0: np.clip(image_rgb.astype(np.float32, copy=False), 0.0, 1.0)}
    candidate_images.update(agss_artifacts.snapshots)

    iteration_log_by_index = {
        int(entry["iteration"]): entry
        for entry in agss_artifacts.summary.get("iterations", [])
    }
    candidate_records = [
        _build_candidate_record(
            iteration,
            candidate_images[iteration],
            rhi_calculator,
            target.value,
            iteration_log_by_index.get(iteration),
        )
        for iteration in selected_candidate_iterations
    ]

    selected_record = min(
        candidate_records,
        key=lambda record: (record["absolute_error"], record["iteration"]),
    )
    selected_iteration = int(selected_record["iteration"])
    output_image = candidate_images[selected_iteration]

    summary = {
        "target_level": target.level,
        "target_sigma": float(target.sigma),
        "target_rhi": float(target.value),
        "selected_iteration": selected_iteration,
        "selected_rhi": float(selected_record["achieved_rhi"]),
        "selected_absolute_error": float(selected_record["absolute_error"]),
        "candidate_iterations": selected_candidate_iterations,
        "candidates": candidate_records,
        "dilation_radius": int(dilation_radius),
        "window_size": int(window_size),
        "selection_rule": "minimum absolute RHI error, tie-broken by smaller iteration",
        "agss": agss_artifacts.summary,
    }

    return E1PartialArtifacts(
        output_image=output_image,
        target_level=target.level,
        target_sigma=float(target.sigma),
        target_rhi=float(target.value),
        selected_iteration=selected_iteration,
        selected_rhi=float(selected_record["achieved_rhi"]),
        agss_artifacts=agss_artifacts,
        summary=summary,
    )
