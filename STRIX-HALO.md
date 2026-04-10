# Running on AMD Strix Halo (gfx1151)

Upstream PyTorch ROCm wheels don't include gfx1151. `HSA_OVERRIDE_GFX_VERSION` doesn't work either. You need TheROCk nightly wheels from AMD's gfx1151 index.

## What we did

1. **Nix dev shell** (`flake.nix`) sets required env vars:
   ```
   HSA_ENABLE_SDMA=0      # prevents SDMA hangs/segfaults
   HSA_USE_SVM=0           # prevents SVM-related crashes
   TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1  # flash attention on RDNA
   ```
   Plus `LD_LIBRARY_PATH` entries for libstdc++, libxcb, libGL, glib, zstd.

2. **Two-step dependency install** (`just sync`):
   - `uv sync --no-install-package torch/torchvision/torchaudio/triton` — installs everything except GPU packages
   - `uv pip install --no-deps --index-url https://rocm.nightlies.amd.com/v2/gfx1151/` — overlays pinned gfx1151 torch, torchvision, torchaudio, triton
   - `uv pip install --index-url ...` — installs rocm-sdk-core, rocm-sdk-libraries-gfx1151, rocm

3. **Pinned versions** (in `justfile`):
   - `torch==2.9.1+rocm7.13.0a20260402`
   - `torchvision==0.24.0+rocm7.13.0a20260402`
   - `torchaudio==2.9.0+rocm7.13.0a20260402`
   - `triton==3.5.1+rocm7.13.0a20260402`

4. **`uv run --no-sync`** for all run/lint/fmt commands — prevents uv from syncing to lockfile and overwriting the gfx1151 wheels.

## Why

- `pyproject.toml` pins torch to the rocm6.4 index (for lockfile resolution), but those wheels lack gfx1151 support
- TheROCk gfx1151 wheels depend on `rocm[libraries]` which uv can't resolve, so `--no-deps` is required
- The upstream `triton` package is broken with ROCm torch — must use the ROCm-built triton
- Without `HSA_ENABLE_SDMA=0`, the first GPU operation segfaults (exit code 139)
