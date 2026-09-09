"""Hebrew-to-English translation with a local MLX model.

`mlx_lm` is imported lazily, inside the load call and inside translate(): it
takes seconds to import, and no other subcommand should pay that just to read
the index.
"""

from __future__ import annotations

from typing import Any

from .config import Config

SYSTEM = (
    "You are a professional Hebrew-to-English translator. Translate the "
    "following Hebrew phone call transcript into natural, fluent English. "
    "Keep each line separate, matching the input line-by-line. Output only "
    "the translation, no commentary."
)

# The generation budget scales with the input rather than using a fixed
# number: a budget sized for a short call truncates a long one mid-sentence,
# and one sized for a long call is unnecessary, since generate() stops at the
# model's own end token well before the cap in practice. config.max_tokens
# (300, tuned for a short topic tag) is far too small for this.
TOKENS_PER_INPUT_TOKEN = 2.5
MIN_TOKENS = 400


class TranslateError(Exception):
    """The model could not be loaded, or produced nothing usable."""


class Translator:
    """A loaded model, reused across every transcript in one run."""

    def __init__(self, model: Any, tokenizer: Any, model_path: str, config: Config):
        self.model = model
        self.tokenizer = tokenizer
        self.model_path = model_path
        self.config = config

    @classmethod
    def load(cls, config: Config, model_path: str | None = None) -> Translator:
        path = model_path or config.model_translate
        try:
            from mlx_lm import load
        except ImportError as exc:  # pragma: no cover - environment problem
            raise TranslateError(f"mlx-lm is not available: {exc}") from exc
        try:
            # load() is typed as returning either (model, tokenizer) or
            # (model, tokenizer, config) - the third only with
            # return_config=True, which we never pass. Index rather than
            # unpack so the declared union doesn't have to be narrowed.
            loaded = load(path)
            model, tokenizer = loaded[0], loaded[1]
        except Exception as exc:
            raise TranslateError(f"failed to load model {path}: {exc}") from exc
        return cls(model, tokenizer, path, config)

    def translate(self, transcript: str) -> str:
        from mlx_lm import generate

        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": transcript},
        ]
        # enable_thinking=False matters for Qwen3: as a reasoning model it
        # otherwise burns the whole token budget on a <think> preamble before
        # ever emitting the translation.
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        input_tokens = len(self.tokenizer.encode(transcript))
        max_tokens = min(
            self.config.translate_max_tokens,
            max(MIN_TOKENS, int(input_tokens * TOKENS_PER_INPUT_TOKEN)),
        )
        text = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
        )
        return text.strip()
