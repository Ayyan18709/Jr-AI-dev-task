"""
ConversationSummaryMemory
─────────────────────────
• Keeps a rolling deque of the last N (default 10) Q&A interactions.
• Maintains an LLM-generated rolling summary that is prepended to every prompt.
• Fully thread-safe via threading.Lock.
• MemoryManager is a process-singleton that maps session_id -> memory instance and persists to disk.
"""

import threading
from collections import deque
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional
import os
import json

from app.core.config import get_settings
from app.core.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
class Interaction:
    """Single Q&A pair with UTC timestamp."""

    def __init__(self, question: str, answer: str) -> None:
        self.question = question
        self.answer = answer
        self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, str]:
        return {
            "question": self.question,
            "answer": self.answer,
            "timestamp": self.timestamp,
        }

    def to_text(self) -> str:
        return f"User: {self.question}\nAssistant: {self.answer}"


# ──────────────────────────────────────────────────────────────────────────────
class ConversationSummaryMemory:
    """
    Rolling memory that stores the last ``max_interactions`` Q&A pairs.

    Usage
    -----
    mem = ConversationSummaryMemory(session_id="abc")
    mem.set_llm(llm_instance)          # inject LLM for summarisation
    mem.add_interaction("Hi", "Hello")
    context_str = mem.get_context_string()
    """

    def __init__(
        self,
        max_interactions: Optional[int] = None,
        session_id: str = "default",
        on_change_callback: Optional[Callable] = None,
    ) -> None:
        self.session_id = session_id
        self.max_interactions = max_interactions or settings.MEMORY_MAX_INTERACTIONS
        self._interactions: deque[Interaction] = deque()
        self.summary: str = ""
        self._lock = threading.Lock()
        self._llm = None            # injected via set_llm()
        self._interaction_counter = 0
        self.on_change_callback = on_change_callback

    # ── public API ────────────────────────────────────────────────────────
    def set_llm(self, llm: Any) -> None:
        """Inject an LLM instance (must expose a .generate(prompt, **kw) method)."""
        self._llm = llm

    def add_interaction(self, question: str, answer: str) -> None:
        """Append a Q&A pair. If buffer exceeds max, pop oldest and incorporate into rolling summary."""
        pruned_interaction = None
        with self._lock:
            self._interactions.append(Interaction(question, answer))
            self._interaction_counter += 1
            
            if len(self._interactions) > self.max_interactions:
                pruned_interaction = self._interactions.popleft()

            logger.debug(
                f"[Memory:{self.session_id}] Added interaction "
                f"({len(self._interactions)}/{self.max_interactions})"
            )
        
        if self.on_change_callback:
            self.on_change_callback()

        # Update summary in background if we popped an old interaction
        if pruned_interaction is not None and self._llm is not None:
            threading.Thread(target=self._update_summary_sync, args=(pruned_interaction,)).start()

    def get_context_string(self) -> str:
        """Build the context block injected into every LLM prompt."""
        with self._lock:
            if not self._interactions and not self.summary:
                return ""
            parts: List[str] = []
            if self.summary:
                parts.append(f"[Summary of Previous Conversation]\n{self.summary}")
            if self._interactions:
                parts.append("[Recent History]")
                parts.extend(i.to_text() for i in self._interactions)
            return "\n\n".join(parts)

    def get_interactions(self) -> List[Dict[str, str]]:
        with self._lock:
            return [i.to_dict() for i in self._interactions]

    def clear(self) -> None:
        with self._lock:
            self._interactions.clear()
            self.summary = ""
            self._interaction_counter = 0
        logger.info(f"[Memory:{self.session_id}] Cleared.")
        if self.on_change_callback:
            self.on_change_callback()

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "session_id": self.session_id,
                "summary": self.summary,
                "interaction_count": len(self._interactions),
                "interactions": [i.to_dict() for i in self._interactions],
            }

    def load_from_dict(self, data: Dict[str, Any]) -> None:
        with self._lock:
            self.summary = data.get("summary", "")
            self._interaction_counter = data.get("interaction_count", 0)
            self._interactions.clear()
            for ix in data.get("interactions", []):
                i = Interaction(ix["question"], ix["answer"])
                i.timestamp = ix["timestamp"]
                self._interactions.append(i)


    # ── private ───────────────────────────────────────────────────────────
    def _update_summary_sync(self, pruned_interaction: Interaction) -> None:
        with self._lock:
            current_summary = self.summary

        prompt = (
            "You are progressively summarizing a conversation. "
            "Given the existing summary, update it to include the details of the new interaction. "
            "Keep the summary concise (3-4 sentences).\n\n"
            f"Existing Summary:\n{current_summary if current_summary else 'No previous summary.'}\n\n"
            f"New Interaction to add:\n{pruned_interaction.to_text()}\n\n"
            "Updated Summary:"
        )
        try:
            raw = self._llm.generate(prompt, max_new_tokens=150)
            with self._lock:
                self.summary = raw.strip()
            logger.debug(f"[Memory:{self.session_id}] Rolling summary updated.")
            if self.on_change_callback:
                self.on_change_callback()
        except Exception as exc:
            logger.warning(f"[Memory:{self.session_id}] Summary failed: {exc}")


# ──────────────────────────────────────────────────────────────────────────────
class MemoryManager:
    """
    Process-singleton that maps ``session_id`` -> ``ConversationSummaryMemory``.
    Thread-safe with disk persistence.
    """

    _instance: Optional["MemoryManager"] = None
    _class_lock = threading.Lock()
    PERSIST_FILE = "data/sessions.json"

    def __new__(cls) -> "MemoryManager":
        with cls._class_lock:
            if cls._instance is None:
                obj = super().__new__(cls)
                obj._sessions: Dict[str, ConversationSummaryMemory] = {}
                obj._lock = threading.Lock()
                cls._instance = obj
                obj._load_from_disk()
        return cls._instance

    def _save_to_disk(self) -> None:
        with self._lock:
            data = {sid: mem.to_dict() for sid, mem in self._sessions.items()}
        try:
            os.makedirs(os.path.dirname(self.PERSIST_FILE), exist_ok=True)
            with open(self.PERSIST_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"[MemoryManager] Failed to save memory to disk: {e}")

    def _load_from_disk(self) -> None:
        if not os.path.exists(self.PERSIST_FILE):
            return
        try:
            with open(self.PERSIST_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            for sid, m_data in data.items():
                mem = ConversationSummaryMemory(session_id=sid, on_change_callback=self._save_to_disk)
                mem.load_from_dict(m_data)
                self._sessions[sid] = mem
            logger.info(f"[MemoryManager] Loaded {len(self._sessions)} sessions from disk.")
        except Exception as e:
            logger.error(f"[MemoryManager] Failed to load memory from disk: {e}")

    def get_or_create(self, session_id: str) -> ConversationSummaryMemory:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = ConversationSummaryMemory(
                    session_id=session_id,
                    on_change_callback=self._save_to_disk
                )
                logger.info(f"[MemoryManager] New session: {session_id}")
                # Save dynamically in a non-blocking thread
                threading.Thread(target=self._save_to_disk).start()
            return self._sessions[session_id]

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.info(f"[MemoryManager] Deleted session: {session_id}")
                threading.Thread(target=self._save_to_disk).start()
                return True
            return False

    def list_sessions(self) -> List[str]:
        with self._lock:
            return list(self._sessions.keys())
