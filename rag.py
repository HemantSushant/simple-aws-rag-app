"""
rag.py — Retrieval + generation pipeline using LangChain.

─────────────────────────────────────────────────────────────────────────────
LANGCHAIN CONCEPTS USED IN THIS FILE
─────────────────────────────────────────────────────────────────────────────

1. DOCUMENT
   The universal container for retrieved text in LangChain.
   Every retriever, vector store, and loader produces Document objects.

   Fields:
     page_content : str   — the chunk text
     metadata     : dict  — arbitrary key/value info (source, score, …)

   Why it matters: having a single standard type lets you swap retrievers
   without changing the rest of your pipeline.

2. RETRIEVER
   Any object that takes a string query and returns list[Document].
   The standard interface is:
       docs = retriever.invoke("my question")

   AmazonKnowledgeBaseRetriever wraps the Bedrock KB Retrieve API behind
   this standard interface. Swap it for a Pinecone or OpenSearch retriever
   later without touching anything downstream.

3. CHATMODEL  (ChatAnthropic)
   A model that works in chat format — it receives a list of messages
   (SystemMessage, HumanMessage, AIMessage) and returns an AIMessage.

   All LangChain ChatModels share the same interface. Swap Claude for
   another model by changing one line; the rest of the code stays the same.

4. CHATPROMPTTEMPLATE
   A reusable prompt blueprint with named placeholders.
   You define it once; at runtime LangChain fills in {context} and {question}
   and produces a list of messages ready to send to the model.

   from_messages() accepts:
     ("system", "...")  →  SystemMessage
     ("human",  "...")  →  HumanMessage

5. STROUTPUTPARSER
   Extracts the plain string from the model's AIMessage response.
   Without it, chain.invoke() returns an AIMessage object.
   With it, you get back a plain str — easier to print and pass around.

6. LCEL — LangChain Expression Language
   The | (pipe) operator connects Runnables into a chain.
   Every Runnable (retrievers, prompts, models, parsers) exposes:
     .invoke(input)              — run once, return result
     .stream(input)              — yield tokens as they arrive
     .batch([input1, input2])    — run multiple inputs in parallel

   Because all Runnables share the same interface, a chain built with |
   automatically inherits all three modes.

   Chain built in build_chain():

     {"context": retriever | _format_docs,
      "question": RunnablePassthrough()}
            │
            ▼  builds the two template variables
         prompt                  (ChatPromptTemplate)
            │
            ▼  renders variables → list[ChatMessage]
           llm                   (ChatAnthropic)
            │
            ▼  LLM call → AIMessage
     StrOutputParser()
            │
            ▼  .content → plain str

7. RUNNABLEPASSTHROUGH
   A no-op Runnable that passes its input through unchanged.
   Used to keep the original "question" string in the dict alongside the
   retrieved context, so both are available when the prompt is rendered.
"""

from langchain_aws import AmazonKnowledgeBasesRetriever
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

from config import Config


# ── Prompt ────────────────────────────────────────────────────────────────────
#
# ChatPromptTemplate.from_messages() takes a list of (role, text) tuples.
# Placeholders like {context} and {question} are filled at runtime by LCEL.
#
# This prompt object is defined once at module level and reused across calls —
# creating it is cheap, so there's no need to rebuild it per request.

PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant that answers questions based strictly on the "
        "provided context. Follow these rules:\n\n"
        "1. Answer ONLY from the context — do not use outside knowledge.\n"
        "2. If the context does not contain enough information, say clearly: "
        "\"I don't have enough information in the provided documents to answer that.\"\n"
        "3. When quoting or referencing specific content, mention which source it came from.\n"
        "4. Be concise and direct.",
    ),
    (
        "human",
        "CONTEXT:\n{context}\n\nQUESTION:\n{question}",
    ),
])


# ── Internal helpers ───────────────────────────────────────────────────────────

def _make_retriever(config: Config) -> AmazonKnowledgeBasesRetriever:
    """
    Build a LangChain AmazonKnowledgeBaseRetriever.

    This wraps the Bedrock KB Retrieve API behind LangChain's standard
    Retriever interface (.invoke() → list[Document]).

    Because it implements the Runnable interface, it can be dropped directly
    into an LCEL chain with |, and it supports .stream() and .batch() for free.
    """
    if not config.kb_id:
        raise EnvironmentError("Missing required environment variable: BEDROCK_KB_ID")

    return AmazonKnowledgeBasesRetriever(
        knowledge_base_id=config.kb_id,
        retrieval_config={
            "vectorSearchConfiguration": {"numberOfResults": config.top_k}
        },
        region_name=config.aws_region,
    )


def _make_llm(config: Config) -> ChatAnthropic:
    """
    Build a LangChain ChatAnthropic model.

    ChatAnthropic is a ChatModel — it takes a list of messages and returns
    an AIMessage. All LangChain ChatModels share the same interface, so
    swapping Claude for another model later only requires changing this line.

    Note: the original code used anthropic's thinking={"type": "adaptive"}.
    That's removed here because StrOutputParser expects a simple text response.
    To re-enable thinking, pass thinking={"type": "enabled", "budget_tokens": 1024}
    to ChatAnthropic and replace StrOutputParser with a custom parser that
    skips thinking blocks.
    """
    return ChatAnthropic(
        model=config.claude_model,
        max_tokens=config.max_tokens,
    )


def _format_docs(docs: list[Document]) -> str:
    """
    Convert a list of LangChain Document objects into the context string
    that the prompt expects.

    This function is used as a transformation step inside the LCEL chain:
        retriever | _format_docs
    LangChain automatically wraps plain callables in a RunnableLambda,
    so any function that takes input and returns output can be used with |.
    """
    parts = []
    for i, doc in enumerate(docs, 1):
        # AmazonKnowledgeBaseRetriever stores the raw Bedrock result in metadata.
        # The S3 URI lives at metadata["location"]["s3Location"]["uri"].
        source = (
            doc.metadata
            .get("location", {})
            .get("s3Location", {})
            .get("uri", "unknown")
        )
        score = round(doc.metadata.get("score", 0), 4)
        parts.append(
            f"[Source {i} | score: {score} | file: {source.split('/')[-1]}]\n"
            f"{doc.page_content}"
        )
    return "\n\n" + ("─" * 60 + "\n\n").join(parts)


# ── Public: full LCEL chain ────────────────────────────────────────────────────

def build_chain(config: Config):
    """
    Build and return the complete RAG chain using LCEL.

    This is the "proper LangChain way" — a single composable object that
    retrieves, formats, prompts, and generates in one call:

        answer = chain.invoke("What is the vacation policy?")
        for token in chain.stream("What is the vacation policy?"):
            print(token, end="")

    The chain is also serialisable, which means you can serve it with
    LangServe, trace it with LangSmith, or wrap it in an Agent later.

    Chain anatomy:
        Step 1 — Build template variables (a dict with two keys):
            "context"  : retriever fetches Documents, _format_docs turns them
                         into the context string
            "question" : RunnablePassthrough() forwards the original query
                         string unchanged

        Step 2 — PROMPT fills {context} and {question} into the template,
                 producing [SystemMessage, HumanMessage]

        Step 3 — llm sends the messages to Claude → AIMessage

        Step 4 — StrOutputParser extracts AIMessage.content → plain str
    """
    retriever = _make_retriever(config)
    llm = _make_llm(config)

    chain = (
        {
            "context":  retriever | _format_docs,   # retrieve → format as string
            "question": RunnablePassthrough(),       # pass original question through
        }
        | PROMPT             # fill template → [SystemMessage, HumanMessage]
        | llm                # call Claude → AIMessage
        | StrOutputParser()  # extract .content → plain str
    )
    return chain


# ── Public: step-by-step API (backward-compatible with main.py) ───────────────
#
# The functions below keep the same signatures as before so main.py needs
# no changes. Internally they use the LangChain primitives above.

def retrieve(question: str, config: Config) -> list[dict]:
    """
    Query Bedrock Knowledge Base via LangChain retriever.

    Returns list[dict] (same shape as before) so the rest of the codebase
    doesn't need to know about LangChain Document objects.

    Internally:
        retriever.invoke(question) → list[Document]
        each Document → {"content", "source", "score"} dict
    """
    retriever = _make_retriever(config)

    # .invoke() is the standard Runnable call — works on all LangChain objects.
    docs: list[Document] = retriever.invoke(question)

    return [
        {
            "content": doc.page_content,
            "source": (
                doc.metadata
                .get("location", {})
                .get("s3Location", {})
                .get("uri", "unknown")
            ),
            "score": round(doc.metadata.get("score", 0), 4),
        }
        for doc in docs
    ]


def build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks (dicts) into a readable context block."""
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(
            f"[Source {i} | score: {c['score']} | file: {c['source'].split('/')[-1]}]\n"
            f"{c['content']}"
        )
    return "\n\n" + ("─" * 60 + "\n\n").join(parts)


def generate(question: str, chunks: list[dict], config: Config) -> str:
    """
    Generate an answer from pre-retrieved chunks using LangChain.

    Uses a partial LCEL chain (prompt | llm | parser) — no retriever here
    because the chunks were already fetched by retrieve().

    chain.stream() yields string tokens as Claude produces them, giving the
    same real-time streaming experience as before.
    """
    llm = _make_llm(config)

    # Partial chain: skip the retriever, inject context directly.
    # PROMPT | llm | StrOutputParser() is itself a valid Runnable.
    chain = PROMPT | llm | StrOutputParser()

    context = build_context(chunks)

    # .stream() on any LCEL chain yields incremental output tokens.
    # This works because every Runnable in the chain propagates streaming.
    answer_parts = []
    for token in chain.stream({"context": context, "question": question}):
        print(token, end="", flush=True)
        answer_parts.append(token)
    print()  # newline after streamed response

    return "".join(answer_parts)


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
    # 1. Retrieve (using LangChain retriever under the hood)
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

    # 3. Generate (using LangChain partial chain)
    return generate(question, chunks, config)


# ── Convenience: pretty-print retrieved sources ───────────────────────────────

def show_sources(chunks: list[dict]) -> None:
    print("\nSources used:")
    for i, c in enumerate(chunks, 1):
        print(f"  {i}. {c['source']} (score: {c['score']})")
