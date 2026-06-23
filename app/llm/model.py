"""
Qwen2.5-1.5B-Instruct LLM wrapper
───────────────────────────────────
• Loads model via HuggingFace Transformers.
• Supports CPU / CUDA / MPS with automatic device detection.
• Thread-safe inference via threading.Lock.
• Exposes both .generate() (raw text) and .chat() (messages list).
"""

import threading
import time
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.core.config import get_settings
from app.core.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


class QwenLLM:
    """
    Thin, production-grade wrapper around Qwen2.5-1.5B-Instruct.

    Example
    -------
    llm = QwenLLM()
    llm.load()
    answer = llm.generate("What is RAG?")
    """

    def __init__(self) -> None:
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.device: str = "cpu"
        self._lock = threading.Lock()
        self._loaded = False
        self._load_elapsed: float = 0.0

    # ── loading ───────────────────────────────────────────────────────────
    def load(self) -> None:
        """Download (if needed) and load model + tokenizer."""
        if self._loaded:
            return

        t0 = time.perf_counter()
        logger.info(f"[LLM] Loading {settings.LLM_MODEL_NAME} ...")

        # Resolve device
        self.device = self._resolve_device()
        logger.info(f"[LLM] Target device: {self.device}")

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.LLM_MODEL_NAME,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Model dtype
        dtype = (
            torch.float16
            if self.device in ("cuda", "mps")
            else torch.float32
        )

        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "torch_dtype": dtype,
        }

        # 4-bit quantisation (CUDA only)
        if settings.LLM_LOAD_IN_4BIT and self.device == "cuda":
            try:
                from transformers import BitsAndBytesConfig  # noqa: PLC0415

                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                model_kwargs["device_map"] = "auto"
                logger.info("[LLM] 4-bit quantisation enabled.")
            except ImportError:
                logger.warning("[LLM] bitsandbytes not available; skipping 4-bit.")

        self.model = AutoModelForCausalLM.from_pretrained(
            settings.LLM_MODEL_NAME, **model_kwargs
        )

        if "device_map" not in model_kwargs:
            self.model = self.model.to(self.device)

        self.model.eval()
        self._loaded = True
        self._load_elapsed = time.perf_counter() - t0
        logger.info(f"[LLM] Ready in {self._load_elapsed:.1f}s")

    # ── inference ─────────────────────────────────────────────────────────
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        """
        Plain text generation – wraps the prompt in the model's chat template.

        Returns
        -------
        str : The assistant's reply (stripped of special tokens).
        """
        if not self._loaded:
            raise RuntimeError("Call llm.load() before generating.")

        max_new_tokens = max_new_tokens or settings.LLM_MAX_NEW_TOKENS
        temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE
        top_p = top_p if top_p is not None else settings.LLM_TOP_P

        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, max_new_tokens, temperature, top_p)

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        """
        Multi-turn chat using the model's built-in chat template.

        Parameters
        ----------
        messages : list of {"role": "user"|"assistant"|"system", "content": str}
        """
        if not self._loaded:
            raise RuntimeError("Call llm.load() before generating.")

        max_new_tokens = max_new_tokens or settings.LLM_MAX_NEW_TOKENS
        temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE
        top_p = top_p if top_p is not None else settings.LLM_TOP_P

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(
            [text], return_tensors="pt", padding=True
        ).to(self.device)

        with self._lock:
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

        # Strip the input tokens from the output
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    # ── helpers ───────────────────────────────────────────────────────────
    @staticmethod
    def _resolve_device() -> str:
        if settings.LLM_DEVICE != "auto":
            return settings.LLM_DEVICE
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def load_time(self) -> float:
        return self._load_elapsed
