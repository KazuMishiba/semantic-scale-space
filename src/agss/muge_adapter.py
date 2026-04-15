from __future__ import annotations

import importlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Independent, Normal


Array = np.ndarray


@dataclass
class _MuGEArgs:
    distribution: str = "gs"


class UAEDMuGEAdapter:
    """Thin adapter for the external `ZhouCX117/UAED_MuGE` repository.

    This class does not copy or modify MuGE source files. It only imports the
    external repository from a user-provided path at runtime.
    """

    def __init__(self, repo_path: str | Path, checkpoint_path: str | Path, device: str | None = "auto") -> None:
        self.repo_path = Path(repo_path).expanduser().resolve()
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.device = self._select_device(device)
        self._model: torch.nn.Module | None = None
        self._validate_inputs()

    PAD_MULTIPLE = 32

    @staticmethod
    def _select_device(device: str | None) -> torch.device:
        if device in (None, "", "auto"):
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _validate_inputs(self) -> None:
        if not self.repo_path.exists():
            raise FileNotFoundError(f"MuGE repo path not found: {self.repo_path}")
        expected_module = self.repo_path / "models" / "sigma_logit_unetpp_alpha_ffthalf_feat.py"
        if not expected_module.exists():
            raise FileNotFoundError(
                "Expected MuGE module was not found. "
                f"Checked: {expected_module}"
            )
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"MuGE checkpoint not found: {self.checkpoint_path}")

    def prepare(self) -> dict[str, Any]:
        start = time.time()
        model = self._load_model()
        init_sec = time.time() - start
        parameter = next(model.parameters())
        return {
            "model_init_sec": init_sec,
            "model_device": str(parameter.device),
            "requested_device": str(self.device),
        }

    def _ensure_import_path(self) -> None:
        repo_str = str(self.repo_path)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

    def _load_model(self) -> torch.nn.Module:
        if self._model is not None:
            return self._model

        self._ensure_import_path()
        model_module = importlib.import_module("models.sigma_logit_unetpp_alpha_ffthalf_feat")
        model_class = getattr(model_module, "Mymodel")

        model = model_class(args=_MuGEArgs())
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        state_dict: dict[str, Any]
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        self._model = model
        return model

    @staticmethod
    def _pad_to_multiple(tensor: torch.Tensor, multiple: int = PAD_MULTIPLE) -> tuple[torch.Tensor, dict[str, int]]:
        _, _, height, width = tensor.shape
        pad_h = (((height - 1) // multiple + 1) * multiple - height)
        pad_w = (((width - 1) // multiple + 1) * multiple - width)
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        can_reflect = (
            height > 1
            and width > 1
            and top < height
            and bottom < height
            and left < width
            and right < width
        )
        pad_mode = "reflect" if can_reflect else "replicate"
        padded = F.pad(tensor, (left, right, top, bottom), mode=pad_mode)
        return padded, {"top": top, "bottom": bottom, "left": left, "right": right}

    def predict_edge_map(self, image_rgb: Array, alpha: float) -> Array:
        model = self._load_model()
        image = np.clip(image_rgb.astype(np.float32, copy=False), 0.0, 1.0)
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        padded, padding = self._pad_to_multiple(image_tensor, multiple=self.PAD_MULTIPLE)

        with torch.no_grad():
            label_bias = torch.ones(1, device=self.device) * float(alpha)
            mean, std = model(padded, label_bias)
            output_dist = Independent(Normal(loc=mean, scale=std + 0.001), 1)
            outputs = torch.sigmoid(output_dist.rsample())

        top = padding["top"]
        bottom = padding["bottom"]
        left = padding["left"]
        right = padding["right"]

        h_end = -bottom if bottom > 0 else None
        w_end = -right if right > 0 else None
        cropped = outputs[:, :, top:h_end, left:w_end]
        return cropped.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
