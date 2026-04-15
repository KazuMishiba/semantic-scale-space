from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


Array = np.ndarray


def load_rgb_image(path: str | Path) -> Array:
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.float32) / 255.0


def save_rgb_image(path: str | Path, image: Array) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(image, 0.0, 1.0)
    Image.fromarray((clipped * 255.0).round().astype(np.uint8), mode="RGB").save(output_path)


def save_grayscale_image(path: str | Path, image: Array) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(image, 0.0, 1.0)
    Image.fromarray((clipped * 255.0).round().astype(np.uint8), mode="L").save(output_path)


def write_json(path: str | Path, payload: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
