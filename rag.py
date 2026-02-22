"""
rag.py — Retrieval + generation pipeline (Option A).

  Bedrock KB  →  retrieve relevant chunks
  Claude API  →  generate answer from those chunks
"""

import boto3
import anthropic

from config import Config


SYSTEM_PROMPT = """You are a helpful assistant that answers questions based strictly on the \
provided context. Follow these rules:

1. Answer ONLY from the context — do not use outside knowledge.
2. If the context does not contain enough information, say clearly:
   "I don't have enough information in the provided documents to answer that."
3. When quoting or referencing specific content, mention which source it came from.
4. Be concise and direct.
"""


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve(question: str, config: Config) -> list[dict]:
    """
    Query Bedrock Knowledge Base and return the top-k relevant chunks.

    Each returned dict has:
      - content  : the chunk text
      - source   : S3 URI of the source file
      - score    : relevance score (higher = more relevant)
    """
    if not config.kb_id:
        raise EnvironmentError("Missing required environment variable: BEDROCK_KB_ID")

    bedrock_runtime = boto3.client(
        "bedrock-agent-runtime",
        region_name=config.aws_region
    )

    response = bedrock_runtime.retrieve(
        knowledgeBaseId=config.kb_id,
        retrievalQuery={"text": question},
        retrievalConfiguration={
            "vectorSearchConfiguration": {"numberOfResults": config.top_k}
        },
    )

    return [
        {
            "content": r["content"]["text"],
            "source": r["location"].get("s3Location", {}).get("uri", "unknown"),
            "score": round(r["score"], 4),
        }
        for r in response["retrievalResults"]
    ]


# ── Generation ────────────────────────────────────────────────────────────────

def build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a readable context block."""
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(
            f"[Source {i} | score: {c['score']} | file: {c['source'].split('/')[-1]}]\n"
            f"{c['content']}"
        )
    return "\n\n" + ("─" * 60 + "\n\n").join(parts)


def generate(question: str, chunks: list[dict], config: Config) -> str:
    """
    Send the retrieved chunks + question to Claude and stream the answer.
    Returns the complete answer text.
    """
    client = anthropic.Anthropic()

    context = build_context(chunks)
    user_message = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"

    with client.messages.stream(
        model=config.claude_model,
        max_tokens=config.max_tokens,
        thinking={"type": "adaptive"},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        # Stream tokens to stdout in real time
        for text in stream.text_stream:
            print(text, end="", flush=True)
        print()  # newline after streamed response

        final = stream.get_final_message()

    # Extract the text block (skip any thinking blocks)
    return next(
        block.text
        for block in final.content
        if block.type == "text"
    )


# ── Full pipeline ─────────────────────────────────────────────────────────────

def ask(question: str, config: Config, verbose: bool = False) -> str:
    """
    End-to-end RAG: retrieve relevant chunks, then generate an answer.

    Args:
        question : the user's question
        config   : RAG configuration
        verbose  : if True, print retrieved chunks before answering

    Returns:
        The generated answer string.
    """
    # 1. Retrieve
    chunks = retrieve(question, config)

    if not chunks:
        return "No relevant documents found in the knowledge base."

    # 2. (Optional) show what was retrieved
    if verbose:
        print(f"\n{'─'*60}")
        print(f"Retrieved {len(chunks)} chunks:")
        for i, c in enumerate(chunks, 1):
            preview = c["content"][:120].replace("\n", " ")
            print(f"  [{i}] score={c['score']}  {preview} …")
        print(f"{'─'*60}\n")

    # 3. Generate
    answer = generate(question, chunks, config)
    return answer


# ── Convenience: pretty-print retrieved sources ───────────────────────────────

def show_sources(chunks: list[dict]) -> None:
    print("\nSources used:")
    for i, c in enumerate(chunks, 1):
        print(f"  {i}. {c['source']} (score: {c['score']})")
