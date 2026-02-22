"""
chunkers.py — Five chunking strategies for RAG ingestion.

Each chunker returns a list of Chunk objects.  The Chunk carries:
  - content      : the text to embed and store in S3
  - metadata     : strategy name, source filename, chunk index, and any
                   strategy-specific extras (headers, parent index, …)
"""

import re
import json

import boto3
import numpy as np
import tiktoken
from dataclasses import dataclass, field


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    content: str
    metadata: dict = field(default_factory=dict)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_encoder():
    return tiktoken.get_encoding("cl100k_base")


def _token_count(text: str, enc) -> int:
    return len(enc.encode(text))


# ── Strategy 1: Fixed-size (token-based) ─────────────────────────────────────

class FixedSizeChunker:
    """
    Split every N tokens regardless of content boundaries.
    Use for: raw unstructured text, logs, OCR output.
    Pros: predictable chunk sizes, fast.
    Cons: can cut mid-sentence; avoid for human-readable prose.
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.enc = _get_encoder()

    def chunk(self, text: str, source: str = "") -> list[Chunk]:
        tokens = self.enc.encode(text)
        chunks = []
        idx = 0
        i = 0

        while i < len(tokens):
            chunk_tokens = tokens[i: i + self.chunk_size]
            content = self.enc.decode(chunk_tokens)
            chunks.append(Chunk(
                content=content,
                metadata={
                    "source": source,
                    "strategy": "fixed_size",
                    "chunk_index": idx,
                    "token_start": i,
                    "token_end": i + len(chunk_tokens),
                }
            ))
            i += self.chunk_size - self.overlap
            idx += 1

        return chunks


# ── Strategy 2: Recursive splitting ──────────────────────────────────────────

class RecursiveChunker:
    """
    Tries to split on progressively smaller boundaries:
      paragraph → newline → sentence → word → character
    Only falls back to the next separator when the chunk is still too large.

    Use for: general articles, PDFs, web pages, reports.
    This is the best default choice for mixed content.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 50,
        separators: list[str] | None = None,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
        self.enc = _get_encoder()

    def _word_count(self, text: str) -> int:
        return len(text.split())

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        if not separators:
            return [text]

        sep = separators[0]
        remaining = separators[1:]

        parts = text.split(sep) if sep else list(text)
        good_splits: list[str] = []
        current = ""

        for part in parts:
            candidate = (current + sep + part).strip() if current else part.strip()

            if self._word_count(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    good_splits.append(current)
                # Part is still too big — recurse with next separator
                if remaining and self._word_count(part) > self.chunk_size:
                    good_splits.extend(self._split_recursive(part, remaining))
                else:
                    current = part.strip()

        if current:
            good_splits.append(current)

        return good_splits

    def chunk(self, text: str, source: str = "") -> list[Chunk]:
        raw = self._split_recursive(text, self.separators)
        chunks: list[Chunk] = []

        for i, content in enumerate(raw):
            if not content.strip():
                continue

            # Prepend tail of previous chunk to maintain context across boundary
            if i > 0 and self.overlap > 0:
                prev_words = raw[i - 1].split()[-self.overlap:]
                content = " ".join(prev_words) + " " + content

            chunks.append(Chunk(
                content=content.strip(),
                metadata={"source": source, "strategy": "recursive", "chunk_index": i}
            ))

        return chunks


# ── Strategy 3: Structure-based (markdown headers) ───────────────────────────

class StructureChunker:
    """
    Splits on H1 / H2 / H3 markdown headers.
    Each chunk = one logical section; the heading breadcrumb is prepended
    so the retriever always knows which section a passage came from.

    Use for: technical docs, wikis, legal documents, knowledge-base articles.
    """

    HEADER_RE = re.compile(r"(#{1,3} .+)")

    def chunk(self, text: str, source: str = "") -> list[Chunk]:
        parts = self.HEADER_RE.split(text)
        chunks: list[Chunk] = []
        headers = {"h1": "", "h2": "", "h3": ""}
        current_content = ""
        idx = 0

        def flush():
            nonlocal current_content, idx
            if current_content.strip():
                chunks.append(self._make_chunk(current_content, headers, source, idx))
                idx += 1
            current_content = ""

        for part in parts:
            part = part.strip()
            if not part:
                continue

            if re.match(r"^# ", part):
                flush()
                headers = {"h1": part[2:].strip(), "h2": "", "h3": ""}
                current_content = part + "\n"

            elif re.match(r"^## ", part):
                flush()
                headers["h2"] = part[3:].strip()
                headers["h3"] = ""
                current_content = part + "\n"

            elif re.match(r"^### ", part):
                flush()
                headers["h3"] = part[4:].strip()
                current_content = part + "\n"

            else:
                current_content += part + "\n"

        flush()
        return chunks

    def _make_chunk(self, content: str, headers: dict, source: str, idx: int) -> Chunk:
        breadcrumb = " > ".join(v for v in headers.values() if v)
        enriched = f"[Section: {breadcrumb}]\n\n{content.strip()}" if breadcrumb else content.strip()
        return Chunk(
            content=enriched,
            metadata={
                "source": source,
                "strategy": "structure",
                "chunk_index": idx,
                "section_h1": headers["h1"],
                "section_h2": headers["h2"],
                "section_h3": headers["h3"],
            }
        )


# ── Strategy 4: Semantic chunking ─────────────────────────────────────────────

class SemanticChunker:
    """
    Splits text where the topic changes by comparing consecutive sentence
    embeddings. When cosine similarity drops below `threshold`, a new
    chunk begins.

    Use for: long-form content where topics shift gradually —
             research papers, books, interview transcripts.
    Note: makes one Bedrock embedding call per sentence — slower and costlier
          than the other strategies. Use when chunk quality matters most.
    """

    SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

    def __init__(self, threshold: float = 0.75, region: str = "us-east-1"):
        self.threshold = threshold
        self.bedrock = boto3.client("bedrock-runtime", region_name=region)

    def _embed(self, text: str) -> np.ndarray:
        # Titan Embeddings v2 — max 8192 tokens input
        resp = self.bedrock.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            body=json.dumps({
                "inputText": text[:8000],
                "dimensions": 1536,
                "normalize": True,     # unit-norm → dot product = cosine sim
            })
        )
        return np.array(json.loads(resp["body"].read())["embedding"])

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))    # vectors already normalized

    def chunk(self, text: str, source: str = "") -> list[Chunk]:
        sentences = [s.strip() for s in self.SENTENCE_RE.split(text) if s.strip()]

        if len(sentences) < 2:
            return [Chunk(content=text, metadata={"source": source, "strategy": "semantic", "chunk_index": 0})]

        print(f"  [semantic] Embedding {len(sentences)} sentences …")
        embeddings = [self._embed(s) for s in sentences]

        chunks: list[Chunk] = []
        current: list[str] = [sentences[0]]
        idx = 0

        for i in range(1, len(sentences)):
            sim = self._cosine_sim(embeddings[i - 1], embeddings[i])
            if sim >= self.threshold:
                current.append(sentences[i])
            else:
                chunks.append(Chunk(
                    content=" ".join(current),
                    metadata={"source": source, "strategy": "semantic", "chunk_index": idx}
                ))
                current = [sentences[i]]
                idx += 1

        if current:
            chunks.append(Chunk(
                content=" ".join(current),
                metadata={"source": source, "strategy": "semantic", "chunk_index": idx}
            ))

        return chunks


# ── Strategy 5: Hierarchical (parent + child) ─────────────────────────────────

class HierarchicalChunker:
    """
    Creates small *child* chunks (precise retrieval) but embeds the full
    *parent* chunk in the stored text so Claude always receives rich context.

    Stored format per file:
        [CONTEXT]
        <parent text — broad surrounding context>

        [RELEVANT SECTION]
        <child text — the specific passage>

    Bedrock KB retrieves the right child; Claude reads both sections.

    Use for: any corpus where you need pinpoint retrieval precision AND
             enough context to actually answer the question.
    """

    def __init__(
        self,
        parent_size: int = 1000,
        child_size: int = 200,
        overlap: int = 20,
    ):
        self.parent_chunker = FixedSizeChunker(chunk_size=parent_size, overlap=0)
        self.child_chunker = FixedSizeChunker(chunk_size=child_size, overlap=overlap)

    def chunk(self, text: str, source: str = "") -> list[Chunk]:
        parents = self.parent_chunker.chunk(text, source)
        result: list[Chunk] = []

        for p_idx, parent in enumerate(parents):
            children = self.child_chunker.chunk(parent.content, source)

            for c_idx, child in enumerate(children):
                enriched = (
                    f"[CONTEXT]\n{parent.content}\n\n"
                    f"[RELEVANT SECTION]\n{child.content}"
                )
                result.append(Chunk(
                    content=enriched,
                    metadata={
                        "source": source,
                        "strategy": "hierarchical",
                        "parent_index": p_idx,
                        "child_index": c_idx,
                        "chunk_index": len(result),
                    }
                ))

        return result


# ── Factory ───────────────────────────────────────────────────────────────────

def get_chunker(config):
    """Return the right chunker based on config.chunk_strategy."""
    from config import ChunkStrategy

    strategy = config.chunk_strategy

    if strategy == ChunkStrategy.FIXED_SIZE:
        return FixedSizeChunker(config.chunk_size, config.chunk_overlap)

    if strategy == ChunkStrategy.RECURSIVE:
        return RecursiveChunker(config.chunk_size, config.chunk_overlap)

    if strategy == ChunkStrategy.STRUCTURE:
        return StructureChunker()

    if strategy == ChunkStrategy.SEMANTIC:
        return SemanticChunker(config.semantic_threshold, config.aws_region)

    if strategy == ChunkStrategy.HIERARCHICAL:
        return HierarchicalChunker(config.parent_size, config.child_size, config.chunk_overlap)

    raise ValueError(f"Unknown strategy: {strategy}")
