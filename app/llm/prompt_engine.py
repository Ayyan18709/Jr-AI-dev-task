"""
Prompt engineering for chat and RAG modes.
Builds structured message lists consumed by QwenLLM.chat().
"""

from typing import Dict, List, Optional

from app.core.config import get_settings

settings = get_settings()

# ── System prompts ─────────────────────────────────────────────────────────────

_SYSTEM_CHAT = (
    "You are a helpful, concise, and knowledgeable AI assistant. "
    "Always be truthful. If you don't know something, say so."
)

_SYSTEM_RAG = (
    "You are a knowledgeable AI assistant that answers questions based strictly "
    "on the provided context. "
    "If the answer cannot be found in the context, say: "
    "'I could not find that information in the provided document.' "
    "Do not hallucinate. Cite the most relevant part of the context when helpful."
)


# ── Public builders ────────────────────────────────────────────────────────────

def build_chat_messages(
    query: str,
    memory_context: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Build message list for general-chat mode.

    Parameters
    ----------
    query          : Current user question.
    memory_context : Optional conversation history string from MemoryManager.
    """
    content = query
    if memory_context:
        content = f"{memory_context}\n\n[Current Question]\n{query}"

    return [
        {"role": "system", "content": _SYSTEM_CHAT},
        {"role": "user", "content": content},
    ]


def build_rag_messages(
    query: str,
    context_chunks: List[str],
    memory_context: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Build message list for RAG mode with injected context chunks.

    Parameters
    ----------
    query          : Current user question.
    context_chunks : Retrieved document chunks (ordered by relevance).
    memory_context : Optional conversation history string from MemoryManager.
    """
    context_block = "\n\n---\n\n".join(
        f"[Chunk {i + 1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
    )

    user_content_parts = []
    if memory_context:
        user_content_parts.append(memory_context)
    user_content_parts.append(f"[Retrieved Context]\n{context_block}")
    user_content_parts.append(f"[Question]\n{query}")

    return [
        {"role": "system", "content": _SYSTEM_RAG},
        {"role": "user", "content": "\n\n".join(user_content_parts)},
    ]


def build_eval_messages(question: str, context: str) -> List[Dict[str, str]]:
    """Prompt template used by the evaluation pipeline."""
    return [
        {
            "role": "system",
            "content": (
                "You are an evaluation assistant. Answer the question using only "
                "the provided context. Be concise."
            ),
        },
        {
            "role": "user",
            "content": f"[Context]\n{context}\n\n[Question]\n{question}",
        },
    ]
