from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer
from llm import LLM
from experiment import SELF_INSTRUCTIONS


def _find_cot_regions(
    token_ids: list[int],
    tokenizer: Any,
    cot_tags: list[tuple[str, str]] | None = None,
) -> list[tuple[int, int]]:
    """
    Find token index ranges that fall within CoT reasoning tags.

    Args:
        token_ids: List of token IDs
        tokenizer: Tokenizer instance
        cot_tags: List of (start_tag, end_tag) tuples, e.g.,
            [("<think>", "</think>")]. If None, uses default "<think>" tags.

    Returns:
        List of (start_idx, end_idx) tuples indicating CoT regions in
        token space. Regions are inclusive: tokens[start_idx:end_idx+1]
        are inside CoT.
    """
    import re

    if cot_tags is None:
        cot_tags = [("<think>", "</think>")]

    # Decode full sequence to text
    full_text = tokenizer.decode(token_ids, skip_special_tokens=False)

    # Find all CoT tag pairs in the text
    char_regions = []
    for start_tag, end_tag in cot_tags:
        # Escape special regex characters
        start_pattern = re.escape(start_tag)
        end_pattern = re.escape(end_tag)

        # Find all pairs (non-greedy matching)
        pattern = f"{start_pattern}(.*?){end_pattern}"
        for match in re.finditer(pattern, full_text, re.DOTALL):
            char_regions.append((match.start(), match.end()))

    if not char_regions:
        return []  # No CoT regions found

    # Map character positions to token indices
    # Strategy: decode tokens incrementally and track character positions
    token_char_positions = []  # List of (start_char, end_char) for each token
    current_pos = 0

    for token_id in token_ids:
        token_text = tokenizer.decode([token_id], skip_special_tokens=False)
        token_start = current_pos
        token_end = current_pos + len(token_text)
        token_char_positions.append((token_start, token_end))
        current_pos = token_end

    # For each character region, find overlapping tokens
    token_regions = []
    for char_start, char_end in char_regions:
        token_start_idx = None
        token_end_idx = None

        for i, (tok_start, tok_end) in enumerate(token_char_positions):
            # Token overlaps with CoT region
            if tok_start < char_end and tok_end > char_start:
                if token_start_idx is None:
                    token_start_idx = i
                token_end_idx = i

        if token_start_idx is not None:
            token_regions.append((token_start_idx, token_end_idx))

    return token_regions


def _find_assistant_response_start(
    token_ids: list[int],
    tokenizer: Any,
) -> int:
    """
    Find the token index where the assistant's response begins.

    Args:
        token_ids: Full conversation token IDs
        tokenizer: Tokenizer instance

    Returns:
        Token index where assistant response starts
    """
    # Decode full sequence
    full_text = tokenizer.decode(token_ids, skip_special_tokens=False)

    # Common patterns for different chat templates
    assistant_markers = [
        "<|im_start|>assistant",
        "<|start_header_id|>assistant<|end_header_id|>",
        "[/INST]",
        "### Assistant:",
        "Assistant:",
    ]

    # Find the last occurrence of any marker (in case of multi-turn)
    marker_pos = -1
    found_marker = None
    for marker in assistant_markers:
        pos = full_text.rfind(marker)
        if pos > marker_pos:
            marker_pos = pos
            found_marker = marker

    if marker_pos == -1:
        # Fallback: assume assistant response is in the second half
        return len(token_ids) // 2

    # Convert character position to token index
    # Decode tokens incrementally to find where marker ends
    current_pos = 0
    for i, token_id in enumerate(token_ids):
        token_text = tokenizer.decode([token_id], skip_special_tokens=False)
        current_pos += len(token_text)

        # Check if we've passed the marker
        if current_pos > marker_pos + len(found_marker):
            # Assistant response starts here or at next token
            return i

    # Fallback
    return len(token_ids) // 2


class DataCollatorForAssistantSupervision:
    """
    Data collator that masks the chat template (system/user turns and padding)
    while supervising the full assistant response. Optional CoT spans surrounded
    by ``cot_tags`` remain masked so that `<think>...</think>` content does not
    contribute to the loss.
    """

    def __init__(
        self,
        tokenizer: Any,
        cot_tags: list[tuple[str, str]] | None = None,
    ):
        self.tokenizer = tokenizer
        self.cot_tags = cot_tags or [("<think>", "</think>")]

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        if not features:
            raise ValueError("Received empty batch in data collator")

        first_feature = features[0]
        if "text" in first_feature:
            texts = [f["text"] for f in features]
            batch_encoding = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
        else:
            features_to_pad: list[dict[str, Any]] = []
            for feature in features:
                processed_feature: dict[str, Any] = {}
                for key, value in feature.items():
                    if key == "labels":
                        continue
                    if isinstance(value, torch.Tensor):
                        processed_feature[key] = value.tolist()
                    else:
                        processed_feature[key] = value
                features_to_pad.append(processed_feature)

            batch_encoding = self.tokenizer.pad(
                features_to_pad,
                padding=True,
                return_tensors="pt",
            )

        input_ids = batch_encoding["input_ids"]
        attention_mask = batch_encoding.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        for i in range(labels.size(0)):
            token_ids_list = input_ids[i].tolist()

            assistant_start = _find_assistant_response_start(
                token_ids_list,
                self.tokenizer,
            )
            if assistant_start > 0:
                labels[i, :assistant_start] = -100

            cot_regions = _find_cot_regions(
                token_ids_list,
                self.tokenizer,
                self.cot_tags,
            )
            for start, end in cot_regions:
                labels[i, start : end + 1] = -100

        batch_encoding["labels"] = labels
        return batch_encoding


@dataclass
class LoRAConfig:
    """Configuration for LoRA/QLoRA fine-tuning."""

    rank: int = 16
    alpha: int = 32
    dropout: float = 0.1
    target_modules: list[str] | str = field(default_factory=lambda: "all-linear")
    use_qlora: bool = True
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    # QLoRA-specific quantization settings
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class TrainingConfig:
    """Configuration for training hyperparameters."""

    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 10
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 3
    optim: str = "paged_adamw_8bit"
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 0.3
    fp16: bool = False
    bf16: bool = False
    max_seq_length: int = 512
    packing: bool = False
    gradient_checkpointing: bool = True

    # CoT (Chain-of-Thought) tag configuration for masking
    cot_tags: list[tuple[str, str]] = field(
        default_factory=lambda: [("<think>", "</think>")]
    )

    # Additional training arguments
    additional_args: dict[str, Any] = field(default_factory=dict)


def _prepare_dataset(
    questions: pd.DataFrame | Sequence[str],
    scores: pd.DataFrame | Sequence[int],
    reasonings: pd.DataFrame | Sequence[str | None] | None,
    instructions: str,
    tokenizer: Any,
) -> Dataset:
    """
    Convert questions and desired responses into a dataset for fine-tuning.

    Args:
        questions: HEXACO questions as DataFrame with 'question' column
            or sequence of strings.
        scores: Response scores as DataFrame with 'score' column
            or sequence of integers (1-5).
        reasonings: Optional reasoning strings as DataFrame with
            'reasoning' column or sequence of strings/None.
        instructions: Instruction prompt to prepend to each question.
        tokenizer: Tokenizer for applying chat template.

    Returns:
        HuggingFace Dataset with 'text' column containing formatted examples.
    """
    # Normalize inputs to lists
    if isinstance(questions, pd.DataFrame):
        question_list = questions['question'].tolist()
    else:
        question_list = list(questions)

    if isinstance(scores, pd.DataFrame):
        score_list = scores['score'].tolist()
    else:
        score_list = list(scores)

    if reasonings is None:
        reasoning_list = [""] * len(question_list)
    elif isinstance(reasonings, pd.DataFrame):
        if 'reasoning' not in reasonings.columns:
            raise ValueError("Reasonings DataFrame must contain 'reasoning' column")
        reasoning_list = reasonings['reasoning'].tolist()
    else:
        reasoning_list = list(reasonings)

    if not (
        len(question_list) == len(score_list) == len(reasoning_list)
    ):
        raise ValueError(
            "Mismatch between question, score, and reasoning counts: "
            f"{len(question_list)}, {len(score_list)}, {len(reasoning_list)}"
        )

    # Check if tokenizer has a chat template
    has_chat_template = hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None

    # Format each example as a conversation
    formatted_examples = []
    for question, score, reasoning in zip(
        question_list,
        score_list,
        reasoning_list,
    ):
        score_int = int(score)
        reasoning_text = "" if reasoning is None else str(reasoning).strip()
        assistant_content = str(score_int)
        if reasoning_text:
            assistant_content = f"{assistant_content} {reasoning_text}"

        conversation = [
            {"role": "user", "content": instructions + str(question)},
            {"role": "assistant", "content": assistant_content},
        ]

        # Apply chat template if available, otherwise use simple fallback
        if has_chat_template:
            formatted_text = tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=False,
            )
        else:
            # Fallback formatting for tokenizers without chat templates
            formatted_text = (
                f"User: {conversation[0]['content']}\n"
                f"Assistant: {conversation[1]['content']}"
                f"{tokenizer.eos_token if tokenizer.eos_token else ''}"
            )

        formatted_examples.append({"text": formatted_text})

    return Dataset.from_list(formatted_examples)


def _load_model_and_tokenizer(
    model_input: LLM | str,
    lora_config: LoRAConfig,
) -> tuple[Any, Any, str]:
    """
    Load model and tokenizer from LLM object or model path.

    Args:
        model_input: Either an LLM object or a model path string.
        lora_config: LoRA configuration including quantization settings.

    Returns:
        Tuple of (model, tokenizer, model_path).
    """
    if isinstance(model_input, LLM):
        # Extract tokenizer and model from LLM object
        tokenizer = model_input.tokenizer
        model = model_input.model
        # Try to infer model path from model config
        model_path = getattr(
            model.config,
            '_name_or_path',
            'unknown_model',
        )
        return model, tokenizer, model_path

    # Load from path string
    model_path = model_input

    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Setup quantization config for QLoRA
    quantization_config = None
    if lora_config.use_qlora:
        compute_dtype = getattr(torch, lora_config.bnb_4bit_compute_dtype)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=lora_config.load_in_4bit,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=lora_config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=lora_config.bnb_4bit_use_double_quant,
        )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Prepare model for k-bit training if using QLoRA
    if lora_config.use_qlora:
        model = prepare_model_for_kbit_training(model)

    return model, tokenizer, model_path


def _setup_lora(
    model: Any,
    lora_config: LoRAConfig,
) -> Any:
    """
    Apply LoRA to the model.

    Args:
        model: The base model to apply LoRA to.
        lora_config: LoRA configuration.

    Returns:
        Model with LoRA adapters applied.
    """
    peft_config = LoraConfig(
        r=lora_config.rank,
        lora_alpha=lora_config.alpha,
        lora_dropout=lora_config.dropout,
        target_modules=lora_config.target_modules,
        bias=lora_config.bias,
        task_type=lora_config.task_type,
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model


def finetune_hexaco(
    questions: pd.DataFrame | Sequence[str],
    scores: pd.DataFrame | Sequence[int],
    reasonings: pd.DataFrame | Sequence[str | None] | None,
    output_path: str,
    model: LLM | str,
    instructions: str = SELF_INSTRUCTIONS,
    lora_config: LoRAConfig | None = None,
    training_config: TrainingConfig | None = None,
) -> None:
    """
    Fine-tune a language model on HEXACO question/response pairs using LoRA/QLoRA.

    This function performs supervised fine-tuning so that the model reproduces
    the desired score and accompanying reasoning when prompted with HEXACO
    questions under the provided instructions.

    Args:
        questions: HEXACO questions as either:
            - DataFrame with a 'question' column, or
            - Sequence of question strings.
        scores: Response scores (1-5) as either:
            - DataFrame with a 'score' column, or
            - Sequence of integers.
        reasonings: Optional reasoning strings paired with each score. Accepts a
            DataFrame with a 'reasoning' column, a sequence of strings, or None
            (treated as empty strings).
        output_path: Directory path where the fine-tuned model/adapter
            will be saved.
        model: Either an LLM object or a string path to a model on disk.
        instructions: Instruction prompt prepended to each question.
            Defaults to SELF_INSTRUCTIONS from experiment.py.
        lora_config: LoRA/QLoRA configuration. If None, uses default
            QLoRA settings (rank=16, 4-bit quantization).
        training_config: Training hyperparameters. If None, uses defaults
            (3 epochs, batch_size=4, lr=2e-4).

    Returns:
        None. The fine-tuned adapter is saved to output_path.

    Raises:
        ValueError: If input lengths do not match.

    Example:
        >>> questions = ["I am the life of the party", "I feel comfortable around people"]
        >>> scores = [4, 5]
        >>> reasons = ["Because I enjoy social settings.", "I feel confident."]
        >>> finetune_hexaco(
        ...     questions=questions,
        ...     scores=scores,
        ...     reasonings=reasons,
        ...     output_path=\"./models/finetuned_adapter\",
        ...     model=\"/path/to/Qwen3-4B\",
        ... )
    """
    # Use default configs if not provided
    if lora_config is None:
        lora_config = LoRAConfig()
    if training_config is None:
        training_config = TrainingConfig()

    # Ensure output directory exists
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model_obj, tokenizer, model_path = _load_model_and_tokenizer(
        model,
        lora_config,
    )

    # Prepare dataset
    print("Preparing dataset...")
    dataset = _prepare_dataset(
        questions,
        scores,
        reasonings,
        instructions,
        tokenizer,
    )
    print(f"Created dataset with {len(dataset)} examples")

    # Apply LoRA
    print("Applying LoRA adapters...")
    model_obj = _setup_lora(model_obj, lora_config)

    # Setup training arguments
    print("Setting up training...")

    # Auto-detect precision based on GPU capabilities
    if training_config.fp16 or training_config.bf16:
        fp16 = training_config.fp16
        bf16 = training_config.bf16
    else:
        # Auto-detect: use bf16 if available (Ampere+ GPUs), else fp16
        bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        fp16 = not bf16

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=training_config.num_train_epochs,
        per_device_train_batch_size=(
            training_config.per_device_train_batch_size
        ),
        gradient_accumulation_steps=(
            training_config.gradient_accumulation_steps
        ),
        learning_rate=training_config.learning_rate,
        warmup_steps=training_config.warmup_steps,
        logging_steps=training_config.logging_steps,
        save_steps=training_config.save_steps,
        save_total_limit=training_config.save_total_limit,
        optim=training_config.optim,
        lr_scheduler_type=training_config.lr_scheduler_type,
        max_grad_norm=training_config.max_grad_norm,
        fp16=fp16,
        bf16=bf16,
        gradient_checkpointing=training_config.gradient_checkpointing,
        report_to=None,
        max_length=training_config.max_seq_length,
        packing=training_config.packing,
        **training_config.additional_args,
    )

    # Create custom data collator that masks prompts but not assistant outputs
    print("Creating data collator for assistant supervision...")
    data_collator = DataCollatorForAssistantSupervision(
        tokenizer=tokenizer,
        cot_tags=training_config.cot_tags,
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model_obj,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save final model
    print(f"Saving model to {output_path}...")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print("Fine-tuning complete!")
