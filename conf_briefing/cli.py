"""CLI entry point for conf-briefing."""

import argparse
import sys

from conf_briefing.config import load_config


def cmd_collect(args):
    """Run the collect pipeline step."""
    from conf_briefing.collect import run_collect

    config = load_config(args.config)
    run_collect(config)


def cmd_clean(args):
    """Run the clean pipeline step."""
    from conf_briefing.clean import run_clean

    config = load_config(args.config)
    run_clean(config)


def cmd_analyze(args):
    """Run the analyze pipeline step."""
    from conf_briefing.analyze import run_analyze

    config = load_config(args.config)
    run_analyze(config)


def cmd_visualize(args):
    """Run the visualize pipeline step."""
    from conf_briefing.visualize import run_visualize

    config = load_config(args.config)
    run_visualize(config)


def cmd_report(args):
    """Run the report pipeline step."""
    from conf_briefing.report import run_report

    config = load_config(args.config)
    run_report(config)


def cmd_index(args):
    """Build the vector index for RAG queries."""
    from conf_briefing.query import run_index

    config = load_config(args.config)
    run_index(config)


def cmd_ask(args):
    """Ask a question about the conference data."""
    from conf_briefing.query import run_ask

    config = load_config(args.config)
    chunk_types = args.chunk_types.split(",") if args.chunk_types else None
    run_ask(
        config,
        question=args.question,
        top_k=args.top_k,
        chunk_types=chunk_types,
        track=args.track,
        verbose=args.verbose,
    )


def cmd_run(args):
    """Run the full pipeline."""
    config = load_config(args.config)

    from conf_briefing.analyze import run_analyze
    from conf_briefing.clean import run_clean
    from conf_briefing.collect import run_collect
    from conf_briefing.report import run_report
    from conf_briefing.visualize import run_visualize

    run_collect(config)
    run_clean(config)
    run_analyze(config)
    run_visualize(config)
    run_report(config)


def main():
    parser = argparse.ArgumentParser(
        prog="conf-briefing",
        description="AI-powered conference analysis tool",
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path to event configuration file (e.g. events/kubecon-eu-2026.toml)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("collect", help="Fetch schedule and transcripts")
    subparsers.add_parser("clean", help="Normalize and structure data")
    subparsers.add_parser("analyze", help="Run LLM analysis")
    subparsers.add_parser("visualize", help="Generate charts and diagrams")
    subparsers.add_parser("report", help="Render report templates")
    subparsers.add_parser("run", help="Run the full pipeline")

    subparsers.add_parser("index", help="Build vector index for RAG queries")

    ask_parser = subparsers.add_parser("ask", help="Ask a question about the conference")
    ask_parser.add_argument("question", help="The question to ask")
    ask_parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=None,
        help="Number of chunks to retrieve (default: from config)",
    )
    ask_parser.add_argument(
        "-t",
        "--chunk-types",
        default=None,
        help="Comma-separated chunk types to filter by",
    )
    ask_parser.add_argument(
        "--track",
        default=None,
        help="Filter by conference track",
    )
    ask_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show retrieved chunks before answer",
    )

    args = parser.parse_args()

    commands = {
        "collect": cmd_collect,
        "clean": cmd_clean,
        "analyze": cmd_analyze,
        "visualize": cmd_visualize,
        "report": cmd_report,
        "run": cmd_run,
        "index": cmd_index,
        "ask": cmd_ask,
    }

    try:
        commands[args.command](args)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
