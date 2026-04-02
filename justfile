set dotenv-load

# List available recipes
default:
    @just --list

# --- Setup ---

# Sync Python dependencies (runs automatically in devshell)
uv-sync:
    uv sync --extra scrape --extra extract

# Sync deps for AMD gfx1151 (Strix Halo) — deps only, no Ollama pull
sync:
    #!/usr/bin/env bash
    set -euo pipefail
    echo ":: Syncing base dependencies (excluding torch packages)..."
    uv sync --extra scrape --extra extract \
        --no-install-package torch \
        --no-install-package torchvision \
        --no-install-package torchaudio \
        --no-install-package triton
    echo ":: Installing WhisperX + transformers..."
    uv pip install --no-deps "whisperx>=3.3" "transformers>=4.40,<5"
    echo ":: Installing gfx1151 PyTorch from TheROCk nightlies..."
    rocm_index="https://rocm.nightlies.amd.com/v2/gfx1151/"
    uv pip install --no-deps \
        --index-url "$rocm_index" \
        "torch==2.9.1+rocm7.13.0a20260402" \
        "torchvision==0.24.0+rocm7.13.0a20260402" \
        "torchaudio==2.10.0+rocm7.13.0a20260325" \
        "triton==3.5.1+rocm7.13.0a20260402"
    echo ":: Installing ROCm SDK runtime..."
    uv pip install --index-url "$rocm_index" \
        rocm-sdk-core rocm-sdk-libraries-gfx1151 rocm
    python3 scripts/fix-execstack.py
    touch .rocm-torch
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
            echo ":: Syncing base dependencies (excluding torch packages)..."
            uv sync $extras \
                --no-install-package torch \
                --no-install-package torchvision \
                --no-install-package torchaudio \
                --no-install-package triton
            echo ":: Installing WhisperX + transformers..."
            # Pin transformers<5 to stay compatible with huggingface-hub in lockfile.
            uv pip install --no-deps "whisperx>=3.3" "transformers>=4.40,<5"
            echo ":: Installing gfx1151 PyTorch from TheROCk nightlies..."
            # Upstream ROCm wheels lack gfx1151 support. TheROCk nightlies
            # depend on rocm[libraries] which uv can't resolve, so --no-deps.
            rocm_index="https://rocm.nightlies.amd.com/v2/gfx1151/"
            uv pip install --no-deps \
                --index-url "$rocm_index" \
                "torch==2.9.1+rocm7.13.0a20260402" \
                "torchvision==0.24.0+rocm7.13.0a20260402" \
                "torchaudio==2.10.0+rocm7.13.0a20260325" \
                "triton==3.5.1+rocm7.13.0a20260402"
            echo ":: Installing ROCm SDK runtime..."
            uv pip install --index-url "$rocm_index" \
                rocm-sdk-core rocm-sdk-libraries-gfx1151 rocm
            # NixOS blocks shared libraries with executable stacks. ctranslate2
            # ships with RWE GNU_STACK — clear the execute bit so it can load.
            python3 scripts/fix-execstack.py
            # Mark that torch was replaced — pipeline recipes use UV_NO_SYNC
            # to prevent uv run from overwriting ROCm torch with CUDA torch.
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

# Extract AI data from collected videos (transcribe, slides, analyze)
extract event:
    {{ _run }} conf-briefing -c events/{{event}}.toml extract

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
