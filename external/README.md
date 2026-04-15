# external

Place external, non-redistributed dependencies here.

Recommended layout:

```text
external/
├── UAED_MuGE/              # clone of https://github.com/ZhouCX117/UAED_MuGE
└── checkpoints/
    └── model_alpha.pth     # checkpoint obtained from the original source
```

Notes:
- Do not commit MuGE source code into this repository.
- Do not commit third-party checkpoints into this repository.
- The Docker / Compose setup mounts this directory into the container at runtime.
