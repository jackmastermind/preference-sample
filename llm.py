from __future__ import annotations

from collections.abc import Callable, Sequence
from threading import Thread
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)

try:
    from torch.backends import mps
except ImportError:
    mps = None  # type: ignore[assignment]


class LLM:
    """Wrapper around huggingface causal language models for chat-style usage."""

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str | None = None,
        device: str | torch.device | None = None,
        model_kwargs: dict[str, Any] | None = None,
        tokenizer_kwargs: dict[str, Any] | None = None,
        generation_defaults: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialise the model and tokenizer.

        Args:
            model_path: Local path or model identifier passed to
                AutoModelForCausalLM.from_pretrained.
            tokenizer_path: Optional tokenizer path. Defaults to model_path.
            device: torch device string or instance. When given, the model is
                moved to this device.
            model_kwargs: Extra keyword arguments for model loading.
            tokenizer_kwargs: Extra keyword arguments for tokenizer loading.
            generation_defaults: Default kwargs forwarded to generate().
        """
        model_kwargs = model_kwargs or {}
        tokenizer_kwargs = tokenizer_kwargs or {}
        tokenizer_location = tokenizer_path or model_path

        self.device = self._resolve_device(device)
        using_device_map = "device_map" in model_kwargs

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_location,
            **tokenizer_kwargs,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs,
        )

        if "padding_side" not in tokenizer_kwargs:
            is_encoder_decoder = getattr(
                self.model.config,
                "is_encoder_decoder",
                False,
            )
            if not is_encoder_decoder:
                self.tokenizer.padding_side = "left"

        if self.device is not None and not using_device_map:
            self.model.to(self.device) # type: ignore

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.history: list[dict[str, str]] = []
        self._generation_defaults = generation_defaults or {
            "max_new_tokens": 256,
            "temperature": 0.7,
        }

    @staticmethod
    def _ensure_attention_mask(
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        if "attention_mask" not in inputs and "input_ids" in inputs:
            input_ids = inputs["input_ids"]
            attention_mask = torch.ones(
                input_ids.shape,
                dtype=torch.long,
                device=input_ids.device,
            )
            inputs = dict(inputs)
            inputs["attention_mask"] = attention_mask
        return inputs

    def reset_history(self) -> None:
        """Clear the stored dialogue history."""
        self.history.clear()

    def chat(
        self,
        prompt: str,
        use_history: bool = True,
        save_history: bool = True,
        stream: bool = False,
        stream_handler: Callable[[str], None] | None = None,
        **generation_kwargs
    ) -> str:
        """
        Run a single chat turn.

        Args:
            prompt: User message for the assistant.
            use_history: Include the stored history in the prompt.
            save_history: Persist this exchange to history.
            stream: Stream generated text chunks via stream_handler.
            stream_handler: Callback for streamed text chunks. Defaults to
                printing to stdout when streaming is enabled.
            generation_kwargs: Overrides for generate().

        Returns:
            The assistant's final response as a decoded string.
        """
        messages = self._build_messages(prompt, use_history)
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        inputs = self._move_to_device(inputs)
        if not isinstance(inputs, torch.Tensor):
            inputs = self._ensure_attention_mask(inputs)

        merged_kwargs = self._merge_generation_kwargs(generation_kwargs)

        if stream:
            handler = stream_handler or self._default_stream_handler
            response_text = self._stream_generate(inputs, merged_kwargs, handler)
        else:
            response_text = self._generate_once(inputs, merged_kwargs)

        cleaned_response = self._clean_generated_text(response_text)

        if save_history:
            self.history.append({"role": "user", "content": prompt})
            self.history.append(
                {"role": "assistant", "content": cleaned_response},
            )

        return cleaned_response

    def batch_generate(
        self,
        prompts: Sequence[str],
        use_history: bool = False,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> list[str]:
        """
        Generate responses for a batch of prompts without altering chat history.

        Args:
            prompts: Iterable of input prompts.
            use_history: Include stored history in each prompt before generation.
            generation_kwargs: Overrides for generate().

        Returns:
            A list of decoded model outputs.
        """
        if not prompts:
            return []

        if use_history:
            message_batches = [
                self._build_messages(prompt, True)
                for prompt in prompts
            ]
        else:
            message_batches = [
                [{"role": "user", "content": prompt}]
                for prompt in prompts
            ]

        inputs = self.tokenizer.apply_chat_template(
            message_batches,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,
        )
        inputs = self._move_to_device(inputs)

        # apply_chat_template with batches returns a tensor, not a dict
        # We need to get attention mask to calculate actual input lengths
        if isinstance(inputs, torch.Tensor):
            attention_mask = (inputs != self.tokenizer.pad_token_id).long()
            prompt_length = inputs.shape[-1]
        else:
            inputs = self._ensure_attention_mask(inputs)
            prompt_length = inputs["input_ids"].shape[-1]

        merged_kwargs = self._merge_generation_kwargs(generation_kwargs)
        with torch.inference_mode():
            if isinstance(inputs, torch.Tensor):
                # Pass attention mask for proper handling of padding
                output_ids = self.model.generate(
                    inputs,
                    attention_mask=attention_mask,
                    **merged_kwargs,
                )
            else:
                output_ids = self.model.generate(
                    **inputs,
                    **merged_kwargs,
                )

        responses: list[str] = []
        for generated in output_ids:
            completion_tokens = generated[prompt_length:]
            if completion_tokens.numel() == 0:
                responses.append("")
                continue
            decoded = self.tokenizer.decode(
                completion_tokens,
                skip_special_tokens=True,
            )
            responses.append(self._clean_generated_text(decoded))

        return responses

    def _merge_generation_kwargs(
        self,
        overrides: dict[str, Any] | None,
    ) -> dict[str, Any]:
        merged = dict(self._generation_defaults)
        if overrides:
            merged.update(overrides)
        return merged

    def _generate_once(
        self,
        inputs: dict[str, Any] | torch.Tensor,
        generation_kwargs: dict[str, Any],
    ) -> str:
        with torch.inference_mode():
            if isinstance(inputs, torch.Tensor):
                attention_mask = torch.ones(
                    inputs.shape,
                    dtype=torch.long,
                    device=inputs.device,
                )
                output_ids = self.model.generate(
                    inputs,
                    attention_mask=attention_mask,
                    **generation_kwargs,
                )
                prompt_length = attention_mask[0].sum().item()
            else:
                inputs = self._ensure_attention_mask(inputs)
                output_ids = self.model.generate(
                    **inputs,
                    **generation_kwargs,
                )
                prompt_length = inputs["attention_mask"][0].sum().item()

        completion_tokens = output_ids[0, prompt_length:]
        decoded = self.tokenizer.decode(
            completion_tokens,
            skip_special_tokens=True,
        )
        return decoded.strip()

    def _stream_generate(
        self,
        inputs: dict[str, Any] | torch.Tensor,
        generation_kwargs: dict[str, Any],
        handler: Callable[[str], None],
    ) -> str:
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        def run_generation() -> None:
            with torch.inference_mode():
                if isinstance(inputs, torch.Tensor):
                    attention_mask = torch.ones(
                        inputs.shape,
                        dtype=torch.long,
                        device=inputs.device,
                    )
                    self.model.generate(
                        inputs,
                        attention_mask=attention_mask,
                        **generation_kwargs,
                        streamer=streamer,
                    )
                else:
                    generate_inputs = self._ensure_attention_mask(inputs)
                    self.model.generate(
                        **generate_inputs,
                        **generation_kwargs,
                        streamer=streamer,
                    )

        thread = Thread(
            target=run_generation,
            daemon=True,
        )
        thread.start()

        collected: list[str] = []
        for token_text in streamer:
            handler(token_text)
            collected.append(token_text)

        thread.join()
        return "".join(collected).strip()

    def _prepare_conversation(
        self,
        prompt: str,
        use_history: bool,
    ) -> str:
        messages = self._build_messages(prompt, use_history)
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

    @staticmethod
    def _default_stream_handler(text_chunk: str) -> None:
        print(text_chunk, end="", flush=True)

    def _move_to_device(
        self,
        inputs: dict[str, Any] | torch.Tensor,
    ) -> dict[str, Any] | torch.Tensor:
        device = self.device or getattr(self.model, "device", None)
        if device is None:
            return inputs

        # Handle bare tensor case
        if isinstance(inputs, torch.Tensor):
            return inputs.to(device)

        # Handle dictionary case
        return {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }

    def get_history(self) -> list[dict[str, str]]:
        """Return a shallow copy of the current chat history."""
        return list(self.history)

    def _build_messages(
        self,
        prompt: str,
        use_history: bool,
    ) -> list[dict[str, str]]:
        if use_history:
            messages = [dict(turn) for turn in self.history]
        else:
            messages = []
        messages.append({"role": "user", "content": prompt})
        return messages

    @staticmethod
    def _clean_generated_text(text: str) -> str:
        if not text:
            return text
        markers = ("<|im_start|>", "<|im_end|>")
        end = len(text)
        for marker in markers:
            idx = text.find(marker)
            if idx != -1:
                end = min(end, idx)
        return text[:end].strip()

    @staticmethod
    def _resolve_device(
        device: str | torch.device | None,
    ) -> torch.device | None:
        if device is not None:
            resolved = torch.device(device)
            if resolved.type == "cuda" and not torch.cuda.is_available():
                raise ValueError("CUDA device requested but not available.")
            if resolved.type == "mps" and (mps is None or not mps.is_available()):
                raise ValueError("MPS device requested but not available.")
            return resolved

        if torch.cuda.is_available():
            return torch.device("cuda")

        if mps is not None and mps.is_available():
            return torch.device("mps")

        return torch.device("cpu")
