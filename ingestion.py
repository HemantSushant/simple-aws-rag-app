"""
ingestion.py — Load documents, chunk them, upload to S3, trigger KB sync.

Flow:
  file on disk  →  load text  →  chunk  →  upload each chunk as .txt to S3
                                          →  start Bedrock KB ingestion job

IMPORTANT: Set your Bedrock Knowledge Base chunking to "None" (no chunking)
so it treats each uploaded file as a single chunk (which is what we want,
since we're doing the chunking ourselves here).
"""

import os
import time
import uuid
import pathlib
from typing import List

import boto3
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.documents import Document

from config import Config
from chunkers import get_chunker


# ── Document loaders ──────────────────────────────────────────────────────────

def load_txt(path: str) -> str:
    loader = TextLoader(path, encoding="utf-8")
    docs = loader.load()
    return "\n\n".join(doc.page_content for doc in docs)


def load_pdf(path: str) -> str:
    loader = PyPDFLoader(path)
    docs = loader.load()
    return "\n\n".join(doc.page_content for doc in docs)


def load_markdown(path: str) -> str:
    # TextLoader handles markdown text just fine for our chunkers
    loader = TextLoader(path, encoding="utf-8")
    docs = loader.load()
    return "\n\n".join(doc.page_content for doc in docs)


def load_document(path: str) -> str:
    """Dispatch to the right loader based on file extension."""
    ext = pathlib.Path(path).suffix.lower()
    loaders = {
        ".txt": load_txt,
        ".pdf": load_pdf,
        ".md":  load_markdown,
        ".markdown": load_markdown,
    }
    if ext not in loaders:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {list(loaders)}")
    return loaders[ext](path)


# ── S3 upload ─────────────────────────────────────────────────────────────────

def upload_chunks_to_s3(
    chunks: List[Document],
    s3_client,
    bucket: str,
    prefix: str = "chunks/",
) -> List[str]:
    """
    Upload each chunk as a separate .txt file to S3.
    Returns the list of S3 object keys that were uploaded.
    """
    keys: List[str] = []

    for chunk in chunks:
        # Unique key: chunks/<source_stem>/<uuid>.txt
        source_stem = pathlib.Path(chunk.metadata.get("source", "doc")).stem
        key = f"{prefix}{source_stem}/{uuid.uuid4().hex}.txt"

        # Store ONLY the plain text in the file body — this is what Bedrock KB
        # embeds and what Claude will read back as context.
        # Metadata goes into S3 object metadata (a separate AWS key-value store)
        # so it never appears in the retrieved text.
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=chunk.page_content.encode("utf-8"),
            ContentType="text/plain",
            Metadata={
                # S3 object metadata values must be strings
                k: str(v)
                for k, v in chunk.metadata.items()
            },
        )
        keys.append(key)

    return keys


# ── Bedrock KB sync ───────────────────────────────────────────────────────────

def start_sync(config: Config) -> str:
    """Trigger a Bedrock KB ingestion job and return the job ID."""
    bedrock_agent = boto3.client("bedrock-agent", region_name=config.aws_region)

    resp = bedrock_agent.start_ingestion_job(
        knowledgeBaseId=config.kb_id,
        dataSourceId=config.ds_id,
    )
    job_id = resp["ingestionJob"]["ingestionJobId"]
    print(f"  KB sync started — job ID: {job_id}")
    return job_id


def wait_for_sync(config: Config, job_id: str, poll_interval: int = 10) -> None:
    """Poll until the ingestion job finishes (or fails)."""
    bedrock_agent = boto3.client("bedrock-agent", region_name=config.aws_region)

    print("  Waiting for KB sync ", end="", flush=True)
    while True:
        resp = bedrock_agent.get_ingestion_job(
            knowledgeBaseId=config.kb_id,
            dataSourceId=config.ds_id,
            ingestionJobId=job_id,
        )
        status = resp["ingestionJob"]["status"]
        print(".", end="", flush=True)

        if status == "COMPLETE":
            print(" done.")
            stats = resp["ingestionJob"].get("statistics", {})
            print(f"  Documents scanned : {stats.get('numberOfDocumentsScanned', '?')}")
            print(f"  Documents indexed : {stats.get('numberOfDocumentsIndexed', '?')}")
            print(f"  Documents failed  : {stats.get('numberOfDocumentsFailed', '?')}")
            return

        if status == "FAILED":
            print(" FAILED.")
            failures = resp["ingestionJob"].get("failureReasons", [])
            raise RuntimeError(f"KB ingestion failed: {failures}")

        time.sleep(poll_interval)


# ── Main ingestion pipeline ───────────────────────────────────────────────────

def ingest(file_path: str, config: Config, sync: bool = True) -> int:
    """
    Full ingestion pipeline for one document:
      1. Load text from file
      2. Chunk with the configured strategy
      3. Upload chunks to S3
      4. (optionally) trigger + await Bedrock KB sync

    Returns the number of chunks created.
    """
    missing = [v for v, val in [("S3_BUCKET", config.s3_bucket), ("BEDROCK_KB_ID", config.kb_id), ("BEDROCK_DS_ID", config.ds_id)] if not val]
    if missing:
        raise EnvironmentError(f"Missing required environment variables for ingest: {', '.join(missing)}")

    source = pathlib.Path(file_path).name
    print(f"\n[ingest] {source}")

    # 1. Load
    print(f"  Loading …")
    text = load_document(file_path)
    print(f"  Loaded {len(text):,} characters")

    # 2. Chunk
    chunker = get_chunker(config)
    print(f"  Chunking with strategy={config.chunk_strategy.value} …")
    chunks = chunker.chunk(text, source=source)
    print(f"  Created {len(chunks)} chunks")

    if not chunks:
        print("  No chunks produced — skipping upload.")
        return 0

    # 3. Upload to S3
    s3 = boto3.client("s3", region_name=config.aws_region)
    print(f"  Uploading to s3://{config.s3_bucket}/chunks/ …")
    keys = upload_chunks_to_s3(chunks, s3, config.s3_bucket)
    print(f"  Uploaded {len(keys)} files")

    # 4. Sync Bedrock KB
    if sync:
        job_id = start_sync(config)
        wait_for_sync(config, job_id)

    return len(chunks)


def ingest_directory(dir_path: str, config: Config, sync_after_all: bool = True) -> int:
    """
    Ingest every supported document in a directory.
    Syncs the KB once at the end (more efficient than per-file syncing).
    """
    supported = {".txt", ".pdf", ".md", ".markdown"}
    files = [
        str(p) for p in pathlib.Path(dir_path).iterdir()
        if p.suffix.lower() in supported
    ]

    if not files:
        print(f"No supported files found in {dir_path}")
        return 0

    total_chunks = 0
    for file_path in files:
        total_chunks += ingest(file_path, config, sync=False)

    if sync_after_all and total_chunks > 0:
        job_id = start_sync(config)
        wait_for_sync(config, job_id)

    print(f"\n[ingest_directory] Done — {len(files)} files, {total_chunks} total chunks")
    return total_chunks
