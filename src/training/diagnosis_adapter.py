from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

from config import ExperimentConfig, resolve_dataset_path, resolve_output_base, resolve_path
from data import build_state_builder
from models import load_model, load_tokenizer


class DiagnosisOnlyDataset(Dataset):
    """Causal-LM SFT dataset whose supervised tokens are only the disease name."""

    def __init__(
        self,
        examples: list[dict[str, Any]],
        tokenizer,
        max_prompt_length: int,
        max_completion_length: int,
    ) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_completion_length = max_completion_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        prompt_ids = self.tokenizer(
            example["prompt"],
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_prompt_length,
        )["input_ids"]

        target_text = str(example["target"]).strip()
        if self.tokenizer.eos_token:
            target_text += self.tokenizer.eos_token
        target_ids = self.tokenizer(
            target_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_completion_length,
        )["input_ids"]

        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "condition": example["condition"],
            "patient_id": example["patient_id"],
        }


class DiagnosisOnlyCollator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(feature["input_ids"]) for feature in features)
        input_ids = torch.full(
            (len(features), max_len),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros((len(features), max_len), dtype=torch.long)
        labels = torch.full((len(features), max_len), -100, dtype=torch.long)

        for idx, feature in enumerate(features):
            length = len(feature["input_ids"])
            input_ids[idx, :length] = torch.tensor(feature["input_ids"], dtype=torch.long)
            attention_mask[idx, :length] = torch.tensor(feature["attention_mask"], dtype=torch.long)
            labels[idx, :length] = torch.tensor(feature["labels"], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class RowwiseCausalLMTrainer(Trainer):
    """Match the memory behavior of fvp-cdm by forwarding one row at a time."""

    def compute_loss(
        self,
        model,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]
        total_label_tokens = (labels != -100).sum().clamp_min(1)

        accumulated_loss = torch.zeros((), device=input_ids.device)
        last_outputs = None
        for row_idx in range(input_ids.size(0)):
            row_labels = labels[row_idx : row_idx + 1]
            row_label_tokens = (row_labels != -100).sum()
            if row_label_tokens.item() == 0:
                continue
            outputs = model(
                input_ids=input_ids[row_idx : row_idx + 1],
                attention_mask=attention_mask[row_idx : row_idx + 1],
                use_cache=False,
            )
            last_outputs = outputs
            shift_logits = outputs.logits[:, :-1, :]
            shift_labels = row_labels[:, 1:]
            supervised_positions = shift_labels != -100
            token_logits = shift_logits[supervised_positions].float()
            token_labels = shift_labels[supervised_positions]
            row_loss = F.cross_entropy(token_logits, token_labels, reduction="mean")
            accumulated_loss = accumulated_loss + row_loss * (row_label_tokens / total_label_tokens)

        if last_outputs is None:
            raise ValueError("No supervised diagnosis tokens found in this batch.")
        return (accumulated_loss, last_outputs) if return_outputs else accumulated_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_output_dir(cfg: ExperimentConfig) -> Path:
    output_dir = resolve_output_base(cfg)
    if cfg.training.timestamp_output_dir:
        output_dir = output_dir / datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_run_info(cfg: ExperimentConfig, output_dir: Path, train_size: int, val_size: int) -> None:
    run_info = {
        "stage": "diagnosis_adapter_sft",
        "config": str(cfg.config_path),
        "output_dir": str(output_dir),
        "model": cfg.model.model_name,
        "train_examples": train_size,
        "val_examples": val_size,
        "target_format": "disease_name_only",
        "next_step": "freeze_adapter_then_train_stop_add_gate_from_final_hidden_state",
    }
    (output_dir / "run_info.json").write_text(
        json.dumps(run_info, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def train_diagnosis_adapter(cfg: ExperimentConfig) -> Path:
    set_seed(cfg.training.seed)
    output_dir = make_output_dir(cfg)

    dataset_root = resolve_path(cfg.root_dir, cfg.data.root_dir)
    builder = build_state_builder(
        builder_name=cfg.data.builder,
        dataset_root=dataset_root,
        diagnosis_prompt_file=resolve_path(cfg.root_dir, cfg.data.diagnosis_prompt_file),
        lab_test_mapping_file=resolve_dataset_path(cfg, cfg.data.lab_test_mapping_file),
    )
    train_examples = builder.build_examples(
        data_file=resolve_dataset_path(cfg, cfg.data.train_file),
        seed=cfg.training.seed,
        max_state_chars=cfg.data.max_state_chars,
        min_visible_tests=cfg.training.min_visible_tests,
        max_visible_tests=cfg.training.max_visible_tests,
        states_per_patient=cfg.training.states_per_patient,
        limit=cfg.data.smoke_limit,
        use_all_available_tests=cfg.training.use_all_available_tests,
    )
    val_examples = builder.build_examples(
        data_file=resolve_dataset_path(cfg, cfg.data.val_file),
        seed=cfg.training.seed + 17,
        max_state_chars=cfg.data.max_state_chars,
        min_visible_tests=cfg.training.min_visible_tests,
        max_visible_tests=cfg.training.max_visible_tests,
        states_per_patient=1,
        limit=cfg.data.smoke_limit,
        use_all_available_tests=cfg.training.use_all_available_tests,
    )

    tokenizer = load_tokenizer(cfg.model.model_name)
    model = load_model(cfg.model, cfg.training)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    train_dataset = DiagnosisOnlyDataset(
        train_examples,
        tokenizer=tokenizer,
        max_prompt_length=cfg.training.max_prompt_length,
        max_completion_length=cfg.training.max_completion_length,
    )
    eval_dataset = DiagnosisOnlyDataset(
        val_examples,
        tokenizer=tokenizer,
        max_prompt_length=cfg.training.max_prompt_length,
        max_completion_length=cfg.training.max_completion_length,
    )

    write_run_info(cfg, output_dir, len(train_dataset), len(eval_dataset))

    args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=cfg.training.num_train_epochs,
        learning_rate=cfg.training.learning_rate,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        bf16=cfg.training.bf16,
        logging_steps=cfg.training.logging_steps,
        save_steps=cfg.training.save_steps,
        eval_strategy="epoch",
        save_strategy="steps",
        report_to=cfg.training.report_to,
        remove_unused_columns=False,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        seed=cfg.training.seed,
    )
    trainer = RowwiseCausalLMTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DiagnosisOnlyCollator(tokenizer),
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    return output_dir
