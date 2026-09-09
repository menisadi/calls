"""Short English summaries of a call, generated from its translation.

Deliberately built on the English translation rather than the Hebrew
transcript, and deliberately using Qwen3-4B rather than DictaLM - a head-to-
head across four combinations (Hebrew/English text x DictaLM/Qwen3-4B) found
the other three each broken in a different way: Qwen3-4B generating Hebrew
directly mixed in stray Chinese characters mid-sentence; DictaLM summarizing
its own English translation stayed accurate on short calls but hallucinated
plausible-sounding fake entity names once a call got long and technical.
Qwen3-4B on the English translation was the only combination that held up
across a short, a dense, and a long call.

`mlx_lm` is imported lazily, inside the load call and inside summarize(): it
takes seconds to import, and no other subcommand should pay that just to read
the index.
"""

from __future__ import annotations

from typing import Any

from .config import Config

SYSTEM = (
    "Give a short summary of the following call transcript, at most 3 "
    "sentences. No bullet lists, no headers, no explanations - just the "
    "summary itself."
)


class SummarizeError(Exception):
    """The model could not be loaded, or produced nothing usable."""


class Summarizer:
    """A loaded model, reused across every translation in one run."""

    def __init__(self, model: Any, tokenizer: Any, model_path: str, config: Config):
        self.model = model
        self.tokenizer = tokenizer
        self.model_path = model_path
        self.config = config

    @classmethod
    def load(cls, config: Config, model_path: str | None = None) -> Summarizer:
        path = model_path or config.model_summarize
        try:
            from mlx_lm import load
        except ImportError as exc:  # pragma: no cover - environment problem
            raise SummarizeError(f"mlx-lm is not available: {exc}") from exc
        try:
            # load() is typed as returning either (model, tokenizer) or
            # (model, tokenizer, config) - the third only with
            # return_config=True, which we never pass. Index rather than
            # unpack so the declared union doesn't have to be narrowed.
            loaded = load(path)
            model, tokenizer = loaded[0], loaded[1]
        except Exception as exc:
            raise SummarizeError(f"failed to load model {path}: {exc}") from exc
        return cls(model, tokenizer, path, config)

    def summarize(self, translation: str) -> str:
        from mlx_lm import generate

        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": translation},
        ]
        # enable_thinking=False matters for Qwen3: as a reasoning model it
        # otherwise burns the whole token budget on a <think> preamble before
        # ever emitting the summary.
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        text = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=self.config.summarize_max_tokens,
            verbose=False,
        )
        return text.strip()
