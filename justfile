# List available recipes
default:
    @just --list

# Default event (override with: just --set event myconf run)
event := "kubecon-eu-2026"

# --- Setup ---

# Sync Python dependencies (runs automatically in devshell)
uv-sync:
    uv sync

# --- Pipeline ---

# Fetch schedule and transcripts
collect:
    uv run conf-briefing -c events/{{event}}.toml collect

# Normalize and structure data
clean:
    uv run conf-briefing -c events/{{event}}.toml clean

# Run LLM analysis
analyze:
    uv run conf-briefing -c events/{{event}}.toml analyze

# Generate charts and diagrams
visualize:
    uv run conf-briefing -c events/{{event}}.toml visualize

# Render report templates
report:
    uv run conf-briefing -c events/{{event}}.toml report

# Run the full pipeline
run:
    uv run conf-briefing -c events/{{event}}.toml run

# --- RAG Query ---

# Build vector index from analysis data
index:
    uv run conf-briefing -c events/{{event}}.toml index

# Ask a question about the conference
ask question:
    uv run conf-briefing -c events/{{event}}.toml ask "{{question}}"

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
