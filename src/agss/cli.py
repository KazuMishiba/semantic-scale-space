from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from .core import run_agss
from .io import load_rgb_image, save_grayscale_image, save_rgb_image, write_json
from .muge_adapter import UAEDMuGEAdapter


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    project_root = _project_root()
    parser = argparse.ArgumentParser(description="Run minimal public AGSS inference on a single image.")
    parser.add_argument("--input", required=True, help="Path to the input RGB image.")
    parser.add_argument("--output", required=True, help="Path to the output image.")
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
    parser.add_argument(
        "--delta-base",
        type=float,
        default=1e-6,
        help="Base Delta-MAD threshold for stage transition.",
    )
    parser.add_argument(
        "--decay",
        type=float,
        default=0.4,
        help="Per-stage threshold decay factor.",
    )
    parser.add_argument("--xi", type=float, default=1e-6, help="Stabilizer added to the AGSS update rule.")
    parser.add_argument("--device", default="auto", help="Torch device, e.g. auto, cuda, cuda:0, cpu.")
    parser.add_argument(
        "--artifacts-dir",
        default=None,
        help="Directory for run metadata and edge maps. Defaults to '<output_stem>_artifacts'.",
    )
    parser.add_argument(
        "--artifacts-level",
        choices=("full", "summary", "none"),
        default="full",
        help="Artifact saving policy: full=summary+edge maps, summary=summary only, none=output image only.",
    )
    parser.add_argument(
        "--skip-edge-map-save",
        action="store_true",
        help="Legacy convenience flag equivalent to omitting edge-map saving while keeping summary output.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    save_artifacts = args.artifacts_level != "none"
    save_edge_maps = args.artifacts_level == "full" and not args.skip_edge_map_save
    artifacts_dir = None
    if save_artifacts:
        artifacts_dir = (
            Path(args.artifacts_dir).expanduser().resolve()
            if args.artifacts_dir
            else output_path.with_name(f"{output_path.stem}_artifacts")
        )

    image_rgb = load_rgb_image(input_path)
    adapter = UAEDMuGEAdapter(
        repo_path=args.muge_repo,
        checkpoint_path=args.muge_checkpoint,
        device=args.device,
    )
    provider_metadata = adapter.prepare()
    artifacts = run_agss(
        image_rgb,
        adapter.predict_edge_map,
        alpha_keys=args.alpha_keys,
        n_iters=args.n_iters,
        radius=args.radius,
        delta_base=args.delta_base,
        decay=args.decay,
        xi=args.xi,
        device=args.device,
        provider_metadata=provider_metadata,
    )

    save_rgb_image(output_path, artifacts.output_image)

    if save_edge_maps and artifacts_dir is not None:
        edge_dir = artifacts_dir / "edge_maps"
        for alpha, edge_map in artifacts.edge_maps.items():
            alpha_token = str(alpha).replace(".", "p")
            save_grayscale_image(edge_dir / f"alpha_{alpha_token}.png", edge_map)

    if save_artifacts and artifacts_dir is not None:
        summary = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "input_image": str(input_path),
            "output_image": str(output_path),
            "artifacts_dir": str(artifacts_dir),
            "artifacts_level": args.artifacts_level,
            "edge_maps_saved": bool(save_edge_maps),
            "muge_repo": str(Path(args.muge_repo).expanduser().resolve()),
            "muge_checkpoint": str(Path(args.muge_checkpoint).expanduser().resolve()),
            "algorithm": "AGSS minimal public baseline",
            "implementation_note": (
                "Current baseline keeps MuGE external while aligning the AGSS core "
                "with the final MAD-difference-based multi-stage algorithm."
            ),
            **artifacts.summary,
        }
        write_json(artifacts_dir / "run_summary.json", summary)
    return 0
