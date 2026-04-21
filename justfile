set dotenv-load

# List available recipes
default:
    @just --list

# --- Setup ---

# Sync Python dependencies (runs automatically in devshell)
uv-sync:
    uv sync --extra scrape --extra extract

# Packages to exclude from uv sync on ROCm — CUDA torch + nvidia runtime libs
_rocm_exclude := "--no-install-package torch --no-install-package torchvision --no-install-package torchaudio --no-install-package triton --no-install-package torchcodec --no-install-package nvidia-cublas-cu12 --no-install-package nvidia-cuda-cupti-cu12 --no-install-package nvidia-cuda-nvrtc-cu12 --no-install-package nvidia-cuda-runtime-cu12 --no-install-package nvidia-cudnn-cu12 --no-install-package nvidia-cufft-cu12 --no-install-package nvidia-cufile-cu12 --no-install-package nvidia-curand-cu12 --no-install-package nvidia-cusolver-cu12 --no-install-package nvidia-cusparse-cu12 --no-install-package nvidia-cusparselt-cu12 --no-install-package nvidia-nccl-cu12 --no-install-package nvidia-nvjitlink-cu12 --no-install-package nvidia-nvtx-cu12"

# Sync deps for AMD gfx1151 (Strix Halo) — deps only, no Ollama pull
sync:
    #!/usr/bin/env bash
    set -euo pipefail
    rocm_index="https://rocm.nightlies.amd.com/v2/gfx1151/"
    echo ":: [1/5] Syncing base dependencies (excluding CUDA packages)..."
    uv sync --extra scrape --extra extract --extra diarize \
        {{ _rocm_exclude }}
    echo ":: [2/5] Installing WhisperX + pyannote-audio + transformers (no-deps)..."
    uv pip install --no-deps "whisperx>=3.3" "pyannote-audio>=3.3" "transformers>=4.40,<5"
    echo ":: [3/5] Installing pyannote-audio runtime deps (no-deps to avoid CUDA torch)..."
    uv pip install --no-deps \
        pyannote.core pyannote.database pyannote.metrics pyannote.pipeline \
        asteroid-filterbanks einops julius lightning lightning-utilities \
        pytorch-metric-learning pytorch-lightning semver tensorboardx \
        torch-audiomentations torch-pitch-shift torchmetrics primePy \
        opentelemetry-exporter-otlp-proto-http opentelemetry-sdk colorlog omegaconf
    echo ":: [4/5] Installing gfx1151 PyTorch + ROCm SDK from TheROCk nightlies..."
    uv pip install --no-deps \
        --index-url "$rocm_index" \
        "torch==2.9.1+rocm7.13.0a20260402" \
        "torchvision==0.24.0+rocm7.13.0a20260402" \
        "torchaudio==2.9.0+rocm7.13.0a20260402" \
        "triton==3.5.1+rocm7.13.0a20260402"
    uv pip install --index-url "$rocm_index" \
        rocm-sdk-core rocm-sdk-libraries-gfx1151 rocm
    echo ":: [5/5] Fixing execstack and verifying..."
    python3 scripts/fix-execstack.py
    touch .rocm-torch
    uv run --no-sync python -c "import torch; assert 'rocm' in torch.__version__, f'Expected ROCm torch, got {torch.__version__}'; print(f'  torch {torch.__version__} OK')"
    echo ":: Done. Run 'just pull-models <event> amd' to also pull Ollama models."

# Sync deps + pull Ollama models (gpu: "cpu", "amd", or "nvidia")
pull-models event="kubecon-eu-2026" gpu="cpu":
    #!/usr/bin/env bash
    set -euo pipefail
    extras="--extra scrape --extra extract"
    case "{{gpu}}" in
        nvidia)
            extras="$extras --extra diarize"
            echo ":: Installing with WhisperX diarization (CUDA)..."
            uv sync $extras
            rm -f .rocm-torch
            ;;
        amd)
            extras="$extras --extra diarize"
            rocm_index="https://rocm.nightlies.amd.com/v2/gfx1151/"
            echo ":: Syncing base dependencies (excluding CUDA packages)..."
            uv sync $extras \
                {{ _rocm_exclude }}
            echo ":: Installing WhisperX + pyannote-audio + transformers (no-deps)..."
            uv pip install --no-deps "whisperx>=3.3" "pyannote-audio>=3.3" "transformers>=4.40,<5"
            echo ":: Installing pyannote-audio runtime deps (no-deps)..."
            uv pip install --no-deps \
                pyannote.core pyannote.database pyannote.metrics pyannote.pipeline \
                asteroid-filterbanks einops julius lightning lightning-utilities \
                pytorch-metric-learning pytorch-lightning semver tensorboardx \
                torch-audiomentations torch-pitch-shift torchmetrics primePy \
                opentelemetry-exporter-otlp-proto-http opentelemetry-sdk colorlog omegaconf
            echo ":: Installing gfx1151 PyTorch + ROCm SDK..."
            uv pip install --no-deps \
                --index-url "$rocm_index" \
                "torch==2.9.1+rocm7.13.0a20260402" \
                "torchvision==0.24.0+rocm7.13.0a20260402" \
                "torchaudio==2.9.0+rocm7.13.0a20260402" \
                "triton==3.5.1+rocm7.13.0a20260402"
            uv pip install --index-url "$rocm_index" \
                rocm-sdk-core rocm-sdk-libraries-gfx1151 rocm
            python3 scripts/fix-execstack.py
            touch .rocm-torch
            ;;
        cpu)
            echo ":: CPU mode — using faster-whisper"
            uv sync $extras
            rm -f .rocm-torch
            ;;
        *)
            echo "Unknown gpu option '{{gpu}}'. Use: cpu, amd, or nvidia"
            exit 1
            ;;
    esac
    uv run --no-sync conf-briefing -c events/{{event}}.toml pull-models

# Helper: run uv with --no-sync when ROCm torch is installed
_run := if path_exists(".rocm-torch") == "true" { "uv run --no-sync" } else { "uv run" }

# --- Pipeline ---

# Collect schedule and recordings for an event
collect event:
    {{ _run }} conf-briefing -c events/{{event}}.toml collect

# Extract data from collected videos (transcribe, slides, clean)
extract event:
    {{ _run }} conf-briefing -c events/{{event}}.toml extract

# Run LLM analysis on extracted data (requires Ollama)
analyze event:
    {{ _run }} conf-briefing -c events/{{event}}.toml analyze

# Remove extract outputs so `just extract` re-processes everything
clean-extract event:
    rm -rf events/{{event}}/slides events/{{event}}/transcripts
    rm -f events/{{event}}/matched.json events/{{event}}/transcripts.json
    @echo "Cleaned extract outputs for {{event}}. Run 'just extract {{event}}' to re-process."

# Generate reports from analyzed data (run extract first)
report event:
    {{ _run }} conf-briefing -c events/{{event}}.toml visualize
    {{ _run }} conf-briefing -c events/{{event}}.toml report

# Interactive Q&A about an event
query event question:
    {{ _run }} conf-briefing -c events/{{event}}.toml index
    {{ _run }} conf-briefing -c events/{{event}}.toml ask "{{question}}"

# Check if extract dependencies are available
extract-check event:
    {{ _run }} python -c "from conf_briefing.extract.preflight import check_extract_ready; from conf_briefing.config import load_config; check_extract_ready(load_config('events/{{event}}.toml'))"

# Build and serve the report as an mdBook (opens browser)
report-book event:
    cd events/{{event}}/reports/book && mdbook serve --open

# Deploy an event's report book to GitHub Pages
deploy-book event:
    bash scripts/deploy-book.sh {{event}}

# --- Docs ---

# Build the mdbook documentation
docs-build:
    cd docs && mdbook build

# Serve the mdbook documentation locally
docs-serve:
    cd docs && mdbook serve --open

# Clean mdbook build output
docs-clean:
    rm -rf docs/book

# --- Dev ---

# Lint and format Python code
lint:
    uv run ruff check conf_briefing/
    uv run ruff format --check conf_briefing/

# Auto-fix lint issues
fix:
    uv run ruff check --fix conf_briefing/
    uv run ruff format conf_briefing/
