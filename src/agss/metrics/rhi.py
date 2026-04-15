from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, gaussian_filter, uniform_filter


Array = np.ndarray


def _load_grayscale_image(path: str | Path) -> Array:
    image = Image.open(path).convert("L")
    return np.asarray(image)


def _normalize_boundary_map(boundary_map: Array) -> Array:
    if boundary_map.ndim != 2:
        raise ValueError(f"Expected HxW boundary map, got {boundary_map.shape}")
    if np.issubdtype(boundary_map.dtype, np.integer):
        return boundary_map > 127
    return boundary_map.astype(np.float32) > 0.5


def _make_disk(radius: int) -> Array:
    if radius < 0:
        raise ValueError("radius must be non-negative")
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (xx * xx + yy * yy) <= radius * radius


def _create_non_edge_mask(boundary_map: Array, dilation_radius: int) -> Array:
    edge_binary = _normalize_boundary_map(boundary_map)
    dilated = binary_dilation(edge_binary, structure=_make_disk(dilation_radius))
    non_edge_mask = ~dilated
    if not np.any(non_edge_mask):
        raise ValueError("Non-edge mask is empty after GT dilation")
    return non_edge_mask


def load_gt_boundary_map(path: str | Path, expected_shape: tuple[int, int] | None = None) -> Array:
    boundary_map = _load_grayscale_image(path)
    if expected_shape is not None and tuple(boundary_map.shape) != tuple(expected_shape):
        raise ValueError(
            f"GT boundary shape mismatch: expected {expected_shape}, got {boundary_map.shape}"
        )
    return boundary_map


@dataclass(frozen=True)
class RhiTarget:
    level: str
    sigma: float
    value: float


class GaussianReferenceRHI:
    """Compute target RHI values from Gaussian-blurred references."""

    DEFAULT_SIGMA_LEVELS = {
        "sigma1": 1.0,
        "sigma2": 2.0,
        "sigma3": 3.0,
    }
    LEVEL_ALIASES = {
        "weak": "sigma1",
        "medium": "sigma2",
        "strong": "sigma3",
    }

    def __init__(
        self,
        input_image: Array,
        gt_boundary: Array | None = None,
        *,
        dilation_radius: int = 6,
        window_size: int = 7,
        sigma_levels: dict[str, float] | None = None,
    ) -> None:
        if input_image.ndim != 3 or input_image.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 RGB image, got {input_image.shape}")
        if window_size <= 0 or window_size % 2 == 0:
            raise ValueError("window_size must be a positive odd integer")

        image = np.clip(input_image.astype(np.float32, copy=False), 0.0, 1.0)
        self.input_image = image
        self.image_shape = image.shape[:2]
        self.dilation_radius = int(dilation_radius)
        self.window_size = int(window_size)
        self.sigma_levels = dict(sigma_levels or self.DEFAULT_SIGMA_LEVELS)

        if gt_boundary is None:
            self.non_edge_mask = np.ones(self.image_shape, dtype=bool)
        else:
            if gt_boundary.shape != self.image_shape:
                raise ValueError(
                    f"GT boundary shape mismatch: expected {self.image_shape}, got {gt_boundary.shape}"
                )
            self.non_edge_mask = _create_non_edge_mask(gt_boundary, self.dilation_radius)

        self.reference_rhi = {
            level: self.compute_rhi(self.input_image, use_gaussian_blur=True, sigma=sigma)
            for level, sigma in self.sigma_levels.items()
        }

    @classmethod
    def from_boundary_path(
        cls,
        input_image: Array,
        gt_boundary_path: str | Path,
        *,
        dilation_radius: int = 6,
        window_size: int = 7,
        sigma_levels: dict[str, float] | None = None,
    ) -> "GaussianReferenceRHI":
        boundary_map = load_gt_boundary_map(gt_boundary_path, expected_shape=input_image.shape[:2])
        return cls(
            input_image,
            boundary_map,
            dilation_radius=dilation_radius,
            window_size=window_size,
            sigma_levels=sigma_levels,
        )

    def _resolve_level(self, level: str) -> str:
        normalized = str(level).strip().lower()
        normalized = self.LEVEL_ALIASES.get(normalized, normalized)
        if normalized not in self.reference_rhi:
            raise ValueError(
                f"Unknown target level '{level}'. Available: {sorted(self.reference_rhi.keys()) + sorted(self.LEVEL_ALIASES.keys())}"
            )
        return normalized

    def _to_linear_luminance(self, image_rgb: Array) -> Array:
        linear_rgb = np.where(
            image_rgb <= 0.04045,
            image_rgb / 12.92,
            np.power((image_rgb + 0.055) / 1.055, 2.4),
        )
        luminance = (
            0.2126 * linear_rgb[:, :, 0]
            + 0.7152 * linear_rgb[:, :, 1]
            + 0.0722 * linear_rgb[:, :, 2]
        )
        return luminance.astype(np.float32, copy=False)

    def _compute_local_variance(self, luminance: Array) -> Array:
        mean = uniform_filter(luminance, size=self.window_size, mode="reflect")
        mean_sq = uniform_filter(luminance * luminance, size=self.window_size, mode="reflect")
        variance = mean_sq - mean * mean
        return np.maximum(variance, 0.0)

    def _apply_gaussian_blur(self, image_rgb: Array, sigma: float) -> Array:
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        blurred = gaussian_filter(image_rgb, sigma=(float(sigma), float(sigma), 0.0), mode="reflect")
        return np.clip(blurred.astype(np.float32, copy=False), 0.0, 1.0)

    def compute_rhi(
        self,
        image_rgb: Array,
        *,
        use_gaussian_blur: bool = False,
        sigma: float | None = None,
    ) -> float:
        image = np.clip(image_rgb.astype(np.float32, copy=False), 0.0, 1.0)
        if image.shape[:2] != self.image_shape:
            raise ValueError(f"Image shape mismatch: expected {self.image_shape}, got {image.shape[:2]}")
        if use_gaussian_blur:
            if sigma is None:
                raise ValueError("sigma must be specified when use_gaussian_blur=True")
            image = self._apply_gaussian_blur(image, float(sigma))
        luminance = self._to_linear_luminance(image)
        local_variance = self._compute_local_variance(luminance)
        return float(np.mean(local_variance[self.non_edge_mask]))

    def get_target_sigma(self, level: str) -> float:
        resolved = self._resolve_level(level)
        return float(self.sigma_levels[resolved])

    def get_target_rhi(self, level: str) -> float:
        resolved = self._resolve_level(level)
        return float(self.reference_rhi[resolved])

    def get_target(self, level: str) -> RhiTarget:
        resolved = self._resolve_level(level)
        return RhiTarget(
            level=resolved,
            sigma=float(self.sigma_levels[resolved]),
            value=float(self.reference_rhi[resolved]),
        )

    def get_all_targets(self) -> dict[str, float]:
        return {level: float(value) for level, value in self.reference_rhi.items()}


__all__ = ["GaussianReferenceRHI", "RhiTarget", "load_gt_boundary_map"]
