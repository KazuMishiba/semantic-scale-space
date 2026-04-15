# CONTAINER

## Purpose

This container setup is intended for local execution of the public AGSS package.

Use this document if you want to run AGSS through Docker / Compose rather than a direct Python environment.

It is not intended to redistribute:
- MuGE source code
- MuGE checkpoints

## Expected host layout

```text
agss/
├── external/
│   ├── UAED_MuGE/
│   └── checkpoints/
│       └── model_alpha.pth
├── docker-compose.yml
└── Dockerfile
```

## Build the container

```bash
docker compose build
```

The compose file is configured for GPU-first execution (`gpus: all`).
It also mounts `./.cache/torch` into `/root/.cache/torch` so that pretrained model downloads are reused across runs.

## Open a shell inside the container

```bash
docker compose run --rm agss-dev
```

## Example inference inside the container

```bash
agss-infer \
  --input /workspace/agss/path/to/input.png \
  --output /workspace/agss/path/to/output.png \
  --n-iters 500 \
  --delta-base 1e-6 \
  --decay 0.4 \
  --xi 1e-6 \
  --artifacts-level summary \
  --device cuda
```

The upstream MuGE model assumes input sizes compatible with multiples of 32.
If the input height or width is not a multiple of 32, the adapter pads it internally before MuGE inference and crops the result back to the original size afterward.

Because the default CLI paths already point to `/workspace/agss/external/...`, no extra MuGE path arguments are needed when the recommended layout is used.

The generated `run_summary.json` for the current implementation uses the parameter names `delta_base`, `decay`, and `xi`.
If only the final image is needed, `--artifacts-level none` can be used.
