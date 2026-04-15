from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from .io import load_rgb_image, save_rgb_image, write_json
from .metrics.rhi import load_gt_boundary_map
from .muge_adapter import UAEDMuGEAdapter
from .pipelines.e1_partial import run_e1_partial_single


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_candidate_iterations(value: str | None) -> list[int] | None:
    if value is None or not value.strip():
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    project_root = _project_root()
    parser = argparse.ArgumentParser(
        description="Run public E1 partial reproduction: target RHI computation and AGSS target matching on a single image."
    )
    parser.add_argument("--input", required=True, help="Path to the input RGB image.")
    parser.add_argument("--gt-boundary", required=True, help="Path to the GT boundary PNG image.")
    parser.add_argument("--output", required=True, help="Path to the selected AGSS output image.")
    parser.add_argument(
        "--summary-path",
        default=None,
        help="Path to the JSON summary. Defaults to '<output_stem>_summary.json'.",
    )
    parser.add_argument(
        "--target-level",
        choices=("sigma1", "sigma2", "sigma3", "weak", "medium", "strong"),
        default="sigma2",
        help="Gaussian reference target level.",
    )
    parser.add_argument(
        "--candidate-iters",
        default=None,
        help="Comma-separated explicit candidate iterations. If omitted, candidate-stride is used.",
    )
    parser.add_argument(
        "--candidate-stride",
        type=int,
        default=5,
        help="Iteration stride for candidate snapshots when candidate-iters is omitted.",
    )
    parser.add_argument(
        "--muge-repo",
        default=str(project_root / "external" / "UAED_MuGE"),
        help="Path to a local clone of ZhouCX117/UAED_MuGE.",
    )
    parser.add_argument(
        "--muge-checkpoint",
        default=str(project_root / "external" / "checkpoints" / "model_alpha.pth"),
        help="Path to the MuGE checkpoint file.",
    )
    parser.add_argument("--alpha-keys", default="3,2,1,0", help="Comma-separated AGSS alpha schedule.")
    parser.add_argument("--n-iters", type=int, default=300, help="Maximum number of smoothing iterations.")
    parser.add_argument("--radius", type=int, default=1, help="Smoothing kernel radius.")
    parser.add_argument("--delta-base", type=float, default=1e-6, help="Base Delta-MAD threshold.")
    parser.add_argument("--decay", type=float, default=0.4, help="Per-stage threshold decay factor.")
    parser.add_argument("--xi", type=float, default=1e-6, help="Stabilizer added to the AGSS update rule.")
    parser.add_argument("--device", default="auto", help="Torch device, e.g. auto, cuda, cuda:0, cpu.")
    parser.add_argument(
        "--dilation-radius",
        type=int,
        default=6,
        help="GT-boundary dilation radius for non-edge mask generation.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=7,
        help="Odd local window size for RHI computation.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    input_path = Path(args.input).expanduser().resolve()
    boundary_path = Path(args.gt_boundary).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    summary_path = (
        Path(args.summary_path).expanduser().resolve()
        if args.summary_path
        else output_path.with_name(f"{output_path.stem}_summary.json")
    )

    image_rgb = load_rgb_image(input_path)
    gt_boundary = load_gt_boundary_map(boundary_path, expected_shape=image_rgb.shape[:2])

    adapter = UAEDMuGEAdapter(
        repo_path=args.muge_repo,
        checkpoint_path=args.muge_checkpoint,
        device=args.device,
    )
    provider_metadata = adapter.prepare()

    artifacts = run_e1_partial_single(
        image_rgb,
        gt_boundary,
        adapter.predict_edge_map,
        target_level=args.target_level,
        candidate_iterations=_parse_candidate_iterations(args.candidate_iters),
        candidate_stride=args.candidate_stride,
        alpha_keys=args.alpha_keys,
        n_iters=args.n_iters,
        radius=args.radius,
        delta_base=args.delta_base,
        decay=args.decay,
        xi=args.xi,
        device=args.device,
        dilation_radius=args.dilation_radius,
        window_size=args.window_size,
        provider_metadata=provider_metadata,
    )

    save_rgb_image(output_path, artifacts.output_image)
    write_json(
        summary_path,
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "input_image": str(input_path),
            "gt_boundary": str(boundary_path),
            "output_image": str(output_path),
            "summary_path": str(summary_path),
            "muge_repo": str(Path(args.muge_repo).expanduser().resolve()),
            "muge_checkpoint": str(Path(args.muge_checkpoint).expanduser().resolve()),
            "algorithm": "AGSS public E1 partial target-RHI matching",
            **artifacts.summary,
        },
    )
    return 0
