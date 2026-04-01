set dotenv-load

# List available recipes
default:
    @just --list

# --- Setup ---

# Sync Python dependencies (runs automatically in devshell)
uv-sync:
    uv sync --extra scrape --extra extract

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
            echo ":: Syncing base dependencies..."
            uv sync $extras
            echo ":: Installing WhisperX + transformers..."
            # Pin transformers<5 to stay compatible with huggingface-hub in lockfile.
            uv pip install "whisperx>=3.3" "transformers>=4.40,<5"
            echo ":: Installing ROCm 7.2 PyTorch (replacing CUDA wheels)..."
            # whisperx pulls CUDA torch as a dependency, so we overwrite it
            # with the ROCm build. --reinstall is required because uv won't
            # replace an installed torch that already satisfies the constraint.
            uv pip install --reinstall torch torchaudio torchvision \
                --index-url https://download.pytorch.org/whl/rocm7.2
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
