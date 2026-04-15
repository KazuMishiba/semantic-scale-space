# REPRODUCIBILITY

## Purpose

This note documents the minimum setup needed to run the public AGSS release accompanying the CVPR 2026 paper *Semantic Scale Space: A Framework for Controllable Image Abstraction*.

This document focuses on a direct Python environment setup. For Docker / Compose usage, see [CONTAINER.md](CONTAINER.md).

## External prerequisites

Prepare the following yourself:
- a clone of `https://github.com/ZhouCX117/UAED_MuGE`
- a MuGE checkpoint obtained from the original distribution
- a Python environment with this package and the MuGE runtime dependencies installed

Recommended placement inside the AGSS checkout:

```text
external/
├── UAED_MuGE/
└── checkpoints/
  └── model_alpha.pth
```

## What this repository does not distribute

This repository does **not** redistribute:
- MuGE source code
- MuGE weights
- SBD or DIV2K datasets

## Recommended setup

1. Clone MuGE somewhere local.
2. Obtain the checkpoint file from the original source.
3. Install a PyTorch build compatible with the local environment.
4. Install this package:
   - `pip install -e .`
5. Ensure the environment also satisfies MuGE runtime imports.

For practical inference, use a CUDA-capable GPU environment.

## Run single-image inference

Example:

```bash
agss-infer \
  --input /path/to/input.png \
  --output /path/to/output.png \
  --alpha-keys 3,2,1,0 \
  --n-iters 500 \
  --radius 1 \
  --delta-base 1e-6 \
  --decay 0.4 \
  --xi 1e-6 \
  --device cuda
```

If you follow the recommended layout above, `--muge-repo` and `--muge-checkpoint` can be omitted.

The main CLI controls are:
- `--alpha-keys`
- `--n-iters`
- `--radius`
- `--delta-base`
- `--decay`
- `--xi`

In JSON summaries and implementation-level parameter names, the corresponding names are recorded as `delta_base`, `decay`, and `xi`.

## Container-based execution

This repository also includes a lightweight Docker / Compose setup for local execution.

For container commands and expected layout, see [CONTAINER.md](CONTAINER.md).

## Produced artifacts

By default, the CLI writes:
- output image
- `<output_stem>_artifacts/run_summary.json`
- `<output_stem>_artifacts/edge_maps/*.png`

Artifact saving can be controlled explicitly:
- `--artifacts-level full`: output image + summary + edge maps
- `--artifacts-level summary`: output image + summary only
- `--artifacts-level none`: output image only

For the current implementation, `run_summary.json` records `delta_base`, `decay`, `xi`, `mad_prev`, `mad_curr`, and `delta_mad`.

## Input-size handling

The public MuGE adapter pads inputs only to the next multiple of 32.
This is the minimum requirement currently supported by the upstream MuGE architecture used here.

Operationally, this means:
- arbitrary input image sizes are accepted
- padding is applied internally only when needed
- MuGE runs on the padded tensor
- outputs are cropped back to the original input size before being written to disk

Example:
- `2040 x 1356` → padded internally to `2048 x 1376` → saved back as `2040 x 1356`

## Known limitations

- The current MuGE adapter expects the upstream MuGE repository to keep the same internal file structure and model entry points as the version used during development.
- This is a minimal public release, not the full research codebase.
- Evaluation code is intentionally out of scope for this step.
