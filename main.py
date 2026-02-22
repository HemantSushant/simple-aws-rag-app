"""
main.py — CLI entry point.

Usage:
  # Ingest a single file
  python main.py ingest --file docs/handbook.pdf --strategy recursive

  # Ingest a whole directory
  python main.py ingest --dir docs/ --strategy hierarchical

  # Interactive Q&A
  python main.py ask

  # Single question (non-interactive)
  python main.py ask --question "What is the vacation policy?"

  # Show retrieved chunks alongside the answer
  python main.py ask --question "..." --verbose

Environment variables required:
  ANTHROPIC_API_KEY   your Anthropic key
  S3_BUCKET           S3 bucket name (documents land here)
  BEDROCK_KB_ID       Bedrock Knowledge Base ID
  BEDROCK_DS_ID       Bedrock Data Source ID (within the KB)
  AWS_REGION          (optional, default: us-east-1)
"""

import argparse
import sys

from config import Config, ChunkStrategy
from ingestion import ingest, ingest_directory
from rag import ask, retrieve, show_sources


# ── Helpers ───────────────────────────────────────────────────────────────────

STRATEGY_CHOICES = [s.value for s in ChunkStrategy]


def build_config(args) -> Config:
    config = Config()
    if hasattr(args, "strategy") and args.strategy:
        config.chunk_strategy = ChunkStrategy(args.strategy)
    if hasattr(args, "top_k") and args.top_k:
        config.top_k = args.top_k
    return config


# ── Sub-commands ──────────────────────────────────────────────────────────────

def cmd_ingest(args):
    config = build_config(args)

    if args.file:
        n = ingest(args.file, config, sync=not args.no_sync)
        print(f"\nIngested {n} chunks from {args.file}")

    elif args.dir:
        n = ingest_directory(args.dir, config, sync_after_all=not args.no_sync)
        print(f"\nIngested {n} total chunks from {args.dir}")

    else:
        print("Error: provide --file or --dir", file=sys.stderr)
        sys.exit(1)


def cmd_ask(args):
    config = build_config(args)

    if args.question:
        # Single shot
        print(f"\nQuestion: {args.question}\n")
        print("Answer:")
        answer = ask(args.question, config, verbose=args.verbose)

        if args.sources:
            chunks = retrieve(args.question, config)
            show_sources(chunks)

    else:
        # Interactive loop
        print("RAG Q&A — type 'quit' to exit, 'sources' to toggle source display\n")
        show_src = False

        while True:
            try:
                question = input("Question: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye.")
                break

            if not question:
                continue
            if question.lower() in ("quit", "exit", "q"):
                print("Goodbye.")
                break
            if question.lower() == "sources":
                show_src = not show_src
                print(f"Source display: {'ON' if show_src else 'OFF'}")
                continue

            print("\nAnswer:")
            answer = ask(question, config, verbose=args.verbose)

            if show_src:
                chunks = retrieve(question, config)
                show_sources(chunks)

            print()


# ── Argument parser ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="RAG system — Bedrock Knowledge Base + Claude"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── ingest ────────────────────────────────────────────────────────────────
    p_ingest = sub.add_parser("ingest", help="Chunk and upload documents to S3 / Bedrock KB")
    src = p_ingest.add_mutually_exclusive_group()
    src.add_argument("--file", help="Path to a single file (.txt, .pdf, .md)")
    src.add_argument("--dir",  help="Path to a directory of documents")
    p_ingest.add_argument(
        "--strategy",
        choices=STRATEGY_CHOICES,
        default=ChunkStrategy.RECURSIVE.value,
        help=(
            "Chunking strategy:\n"
            "  fixed_size   — split every N tokens (good for logs/OCR)\n"
            "  recursive    — paragraph→sentence→word fallback (default, general purpose)\n"
            "  structure    — split on markdown H1/H2/H3 headers (docs/wikis)\n"
            "  semantic     — embedding-based topic boundaries (papers/books)\n"
            "  hierarchical — small child + large parent context (best precision+context)\n"
        ),
    )
    p_ingest.add_argument(
        "--no-sync",
        action="store_true",
        help="Upload to S3 but do not trigger KB sync (useful for batching)",
    )

    # ── ask ───────────────────────────────────────────────────────────────────
    p_ask = sub.add_parser("ask", help="Ask a question against the knowledge base")
    p_ask.add_argument("--question", "-q", help="Question (omit for interactive mode)")
    p_ask.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve (default: 5)")
    p_ask.add_argument("--verbose", "-v", action="store_true", help="Print retrieved chunks before answering")
    p_ask.add_argument("--sources", "-s", action="store_true", help="Print source filenames after answering")

    args = parser.parse_args()

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "ask":
        cmd_ask(args)


if __name__ == "__main__":
    main()
