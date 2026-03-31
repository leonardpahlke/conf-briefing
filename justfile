# List available recipes
default:
    @just --list

# --- Setup ---

# Sync Python dependencies (runs automatically in devshell)
uv-sync:
    uv sync --extra scrape --extra extract

# Pull required Ollama models for an event
pull-models event="kubecon-eu-2026":
    uv run conf-briefing -c events/{{event}}.toml pull-models

# --- Pipeline ---

# Collect schedule and recordings for an event
collect event:
    uv run conf-briefing -c events/{{event}}.toml collect

# Extract AI data from collected videos (transcribe, slides, analyze)
extract event:
    uv run conf-briefing -c events/{{event}}.toml extract

# Generate reports from collected data
report event:
    uv run conf-briefing -c events/{{event}}.toml clean
    uv run conf-briefing -c events/{{event}}.toml analyze
    uv run conf-briefing -c events/{{event}}.toml visualize
    uv run conf-briefing -c events/{{event}}.toml report

# Interactive Q&A about an event
query event question:
    uv run conf-briefing -c events/{{event}}.toml index
    uv run conf-briefing -c events/{{event}}.toml ask "{{question}}"

# Check if extract dependencies are available
extract-check event:
    uv run python -c "from conf_briefing.extract.preflight import check_extract_ready; from conf_briefing.config import load_config; check_extract_ready(load_config('events/{{event}}.toml'))"

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
