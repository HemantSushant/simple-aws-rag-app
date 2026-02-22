import os
from enum import Enum
from dataclasses import dataclass, field


class ChunkStrategy(Enum):
    FIXED_SIZE = "fixed_size"       # Split every N tokens — simple, fast
    RECURSIVE = "recursive"         # Try paragraph → sentence → word (default)
    STRUCTURE = "structure"         # Split on markdown headers
    SEMANTIC = "semantic"           # Split when topic changes (embedding-based)
    HIERARCHICAL = "hierarchical"   # Small child chunks, large parent context


@dataclass
class Config:
    # ── AWS ──────────────────────────────────────────────────────────────────
    # Use .get() so Config() can be constructed without all vars present.
    # Validation happens in the functions that actually need each value.
    aws_region: str = field(default_factory=lambda: os.environ.get("AWS_REGION", "us-east-1"))
    s3_bucket: str = field(default_factory=lambda: os.environ.get("S3_BUCKET", ""))
    kb_id: str = field(default_factory=lambda: os.environ.get("BEDROCK_KB_ID", ""))
    ds_id: str = field(default_factory=lambda: os.environ.get("BEDROCK_DS_ID", ""))

    # ── Claude ───────────────────────────────────────────────────────────────
    claude_model: str = "claude-opus-4-6"
    max_tokens: int = 1024

    # ── Chunking ─────────────────────────────────────────────────────────────
    chunk_strategy: ChunkStrategy = ChunkStrategy.RECURSIVE
    chunk_size: int = 500       # tokens (fixed/recursive) or words (structure)
    chunk_overlap: int = 50     # token overlap between consecutive chunks
    semantic_threshold: float = 0.75   # cosine sim below this → new chunk
    parent_size: int = 1000    # tokens — hierarchical parent chunk size
    child_size: int = 200      # tokens — hierarchical child chunk size

    # ── Retrieval ────────────────────────────────────────────────────────────
    top_k: int = 5             # number of chunks to retrieve per query
