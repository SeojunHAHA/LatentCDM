from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftModel
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

from config import ExperimentConfig, resolve_dataset_path, resolve_output_base, resolve_path
from data import build_state_builder
from models import load_model, load_tokenizer
from models.latent_bridge import BeliefProjector, KVMemoryBridge


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


def build_planner_examples(
    cfg: ExperimentConfig,
    split_file: str,
    seed: int,
    states_per_patient: int,
    limit: int | None,
) -> list[dict[str, Any]]:
    dataset_root = resolve_path(cfg.root_dir, cfg.data.root_dir)
    builder = build_state_builder(
        builder_name=cfg.data.builder,
        dataset_root=dataset_root,
        diagnosis_prompt_file=resolve_path(cfg.root_dir, cfg.data.diagnosis_prompt_file),
        lab_test_mapping_file=resolve_dataset_path(cfg, cfg.data.lab_test_mapping_file),
    )
    df = pd.read_csv(resolve_dataset_path(cfg, split_file))
    if limit:
        df = df.iloc[:limit].reset_index(drop=True)

    examples = []
    max_visible_tests = min(cfg.training.max_visible_tests, len(builder.all_tests))
    min_visible_tests = max(1, cfg.training.min_visible_tests)
    for row_idx, row in df.iterrows():
        available_tests = builder.available_tests(row)
        if not available_tests:
            continue
        for state_idx in range(states_per_patient):
            rng = random.Random(seed + int(row_idx) * 1009 + state_idx)
            visible_tests = (
                list(available_tests)
                if cfg.training.use_all_available_tests
                else builder.sample_visible_tests(
                    available_tests=available_tests,
                    rng=rng,
                    min_visible_tests=min_visible_tests,
                    max_visible_tests=max_visible_tests,
                )
            )
            example = builder.build_example(
                row=row,
                visible_tests=visible_tests,
                max_state_chars=cfg.data.max_state_chars,
                state_sample_id=state_idx,
            )
            visible_set = set(visible_tests)
            missing_tests = [test for test in builder.all_tests if test in available_tests and test not in visible_set]
            if missing_tests:
                target_action = f"next_test: {missing_tests[0]}"
            else:
                target_action = f"stop: {example['target']}"
            planner_prompt = (
                "You are a clinical planning agent.\n"
                "Use the latent diagnostic memory and the visible clinical state to decide the next action.\n"
                "Return exactly one action in the form `next_test: <test>` or `stop: <diagnosis>`.\n\n"
                f"{example['prompt']}\n\n"
                f"{cfg.planner.action_prefix} "
            )
            examples.append(
                {
                    **example,
                    "planner_prompt": planner_prompt,
                    "target_action": target_action,
                    "available_tests": ", ".join(available_tests),
                    "missing_tests": ", ".join(missing_tests),
                }
            )
    return examples


class PlannerActionDataset(Dataset):
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
        item = self.examples[index]
        prompt_ids = self.tokenizer(
            item["planner_prompt"],
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_prompt_length,
        )["input_ids"]
        target_text = str(item["target_action"]).strip()
        if self.tokenizer.eos_token:
            target_text += self.tokenizer.eos_token
        target_ids = self.tokenizer(
            target_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_completion_length,
        )["input_ids"]
        return {
            "input_ids": prompt_ids + target_ids,
            "attention_mask": [1] * (len(prompt_ids) + len(target_ids)),
            "labels": [-100] * len(prompt_ids) + target_ids,
            "diagnosis_prompt": item["prompt"],
            "patient_id": item["patient_id"],
            "target_action": item["target_action"],
        }


class PlannerCollator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
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
            "diagnosis_prompt": [feature["diagnosis_prompt"] for feature in features],
        }


class LatentPlanner(nn.Module):
    def __init__(
        self,
        diagnosis_model,
        planner_model,
        tokenizer,
        candidate_diseases: list[str],
        memory_tokens_per_layer: int,
        num_belief_tokens: int,
        bridge_hidden_dim: int,
    ) -> None:
        super().__init__()
        self.diagnosis_model = diagnosis_model
        self.planner_model = planner_model
        self.tokenizer = tokenizer
        self.candidate_diseases = candidate_diseases
        cfg = planner_model.config
        planner_dim = cfg.hidden_size
        kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        head_dim = planner_dim // cfg.num_attention_heads
        kv_dim = 2 * kv_heads * head_dim
        self.kv_bridge = KVMemoryBridge(
            num_layers=cfg.num_hidden_layers,
            kv_dim=kv_dim,
            planner_dim=planner_dim,
            hidden_dim=bridge_hidden_dim,
            memory_tokens_per_layer=memory_tokens_per_layer,
        )
        self.belief_projector = BeliefProjector(
            num_diseases=len(candidate_diseases),
            planner_dim=planner_dim,
            hidden_dim=bridge_hidden_dim,
            num_belief_tokens=num_belief_tokens,
        )
        for param in self.diagnosis_model.parameters():
            param.requires_grad = False
        self.diagnosis_model.eval()

    def _encode_prompts(self, prompts: list[str], device: torch.device) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
            add_special_tokens=False,
        )
        return {key: value.to(device) for key, value in encoded.items()}

    @torch.no_grad()
    def _candidate_belief(self, prompts: list[str], device: torch.device) -> torch.Tensor:
        # This is an autoregressive candidate likelihood computed with teacher forcing:
        # each disease token is scored at the position where a causal LM would generate it,
        # conditioned on the prompt and all previous disease tokens.
        scores = []
        for disease in self.candidate_diseases:
            candidate_texts = [prompt + disease for prompt in prompts]
            candidate_token_count = len(
                self.tokenizer(
                    disease,
                    add_special_tokens=False,
                )["input_ids"]
            )
            encoded_full = self.tokenizer(
                candidate_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,
                add_special_tokens=False,
            ).to(device)
            outputs = self.diagnosis_model(
                input_ids=encoded_full["input_ids"],
                attention_mask=encoded_full["attention_mask"],
                use_cache=False,
            )
            logits = outputs.logits[:, :-1, :].float()
            labels = encoded_full["input_ids"][:, 1:]
            batch_scores = []
            for row_idx in range(len(prompts)):
                sequence_length = int(encoded_full["attention_mask"][row_idx].sum().item())
                pad_count = encoded_full["attention_mask"].size(1) - sequence_length
                start = max(pad_count + sequence_length - candidate_token_count - 1, 0)
                valid = encoded_full["attention_mask"][row_idx, 1:].bool()
                positions = torch.arange(labels.size(1), device=device)
                candidate_mask = valid & (positions >= start)
                token_logits = logits[row_idx][candidate_mask]
                token_labels = labels[row_idx][candidate_mask]
                token_log_probs = F.log_softmax(token_logits, dim=-1)
                gathered = token_log_probs.gather(1, token_labels[:, None]).squeeze(1)
                batch_scores.append(gathered.mean() if gathered.numel() else torch.tensor(-100.0, device=device))
            scores.append(torch.stack(batch_scores))
        return torch.softmax(torch.stack(scores, dim=1), dim=1)

    @torch.no_grad()
    def _diagnosis_kv(self, prompts: list[str], device: torch.device):
        encoded = self._encode_prompts(prompts, device)
        outputs = self.diagnosis_model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            use_cache=True,
        )
        return outputs.past_key_values

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        diagnosis_prompt: list[str],
    ):
        device = input_ids.device
        past_key_values = self._diagnosis_kv(diagnosis_prompt, device)
        belief = self._candidate_belief(diagnosis_prompt, device)
        memory_tokens = self.kv_bridge(past_key_values)
        belief_tokens = self.belief_projector(belief)

        prompt_embeds = self.planner_model.get_input_embeddings()(input_ids)
        latent_tokens = torch.cat([belief_tokens, memory_tokens], dim=1).to(prompt_embeds.dtype)
        inputs_embeds = torch.cat([latent_tokens, prompt_embeds], dim=1)
        latent_mask = torch.ones(
            input_ids.size(0),
            latent_tokens.size(1),
            dtype=attention_mask.dtype,
            device=device,
        )
        full_attention_mask = torch.cat([latent_mask, attention_mask], dim=1)
        latent_labels = torch.full(
            (input_ids.size(0), latent_tokens.size(1)),
            -100,
            dtype=labels.dtype,
            device=device,
        )
        full_labels = torch.cat([latent_labels, labels], dim=1)
        return self.planner_model(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=full_labels,
            use_cache=False,
        )


class LatentPlannerTrainer(Trainer):
    """Save only the trainable planner adapter and latent bridge modules."""

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False):
        output_path = Path(output_dir or self.args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        model = self.model
        if hasattr(model.planner_model, "peft_config"):
            model.planner_model.save_pretrained(str(output_path / "planner_lora"))
            model.tokenizer.save_pretrained(str(output_path / "planner_lora"))
        else:
            model.tokenizer.save_pretrained(str(output_path / "planner_tokenizer"))
        torch.save(
            {
                "kv_bridge": model.kv_bridge.state_dict(),
                "belief_projector": model.belief_projector.state_dict(),
                "candidate_diseases": model.candidate_diseases,
                "memory_tokens_per_layer": model.kv_bridge.memory_tokens_per_layer,
                "num_belief_tokens": model.belief_projector.num_belief_tokens,
            },
            output_path / "latent_bridges.pt",
        )


def write_run_info(cfg: ExperimentConfig, output_dir: Path, train_size: int, val_size: int) -> None:
    run_info = {
        "stage": "planner_adapter_latent_bridge",
        "config": str(cfg.config_path),
        "output_dir": str(output_dir),
        "model": cfg.model.model_name,
        "diagnosis_adapter_dir": cfg.planner.diagnosis_adapter_dir,
        "train_examples": train_size,
        "val_examples": val_size,
        "latent_inputs": {
            "kv": "all-layer DA past_key_values pooled through KVMemoryBridge",
            "logits": "candidate sequence likelihood softmax through BeliefProjector",
        },
    }
    (output_dir / "run_info.json").write_text(
        json.dumps(run_info, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def train_planner_adapter(cfg: ExperimentConfig) -> Path:
    if not cfg.planner.diagnosis_adapter_dir:
        raise ValueError("Set planner.diagnosis_adapter_dir to a trained diagnosis adapter checkpoint.")

    set_seed(cfg.training.seed)
    output_dir = make_output_dir(cfg)
    tokenizer = load_tokenizer(cfg.model.model_name)

    train_examples = build_planner_examples(
        cfg,
        split_file=cfg.data.train_file,
        seed=cfg.training.seed,
        states_per_patient=cfg.training.states_per_patient,
        limit=cfg.data.smoke_limit,
    )
    val_examples = build_planner_examples(
        cfg,
        split_file=cfg.data.val_file,
        seed=cfg.training.seed + 17,
        states_per_patient=1,
        limit=cfg.data.smoke_limit,
    )

    diagnosis_base = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_name,
        torch_dtype=torch.bfloat16 if cfg.training.bf16 else torch.float16,
        device_map={"": 0},
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    diagnosis_model = PeftModel.from_pretrained(diagnosis_base, cfg.planner.diagnosis_adapter_dir)
    planner_model = load_model(cfg.model, cfg.training)
    if cfg.model.use_lora and hasattr(planner_model, "print_trainable_parameters"):
        planner_model.print_trainable_parameters()
    if not cfg.model.use_lora:
        for param in planner_model.parameters():
            param.requires_grad = False

    model = LatentPlanner(
        diagnosis_model=diagnosis_model,
        planner_model=planner_model,
        tokenizer=tokenizer,
        candidate_diseases=cfg.planner.candidate_diseases,
        memory_tokens_per_layer=cfg.planner.memory_tokens_per_layer,
        num_belief_tokens=cfg.planner.num_belief_tokens,
        bridge_hidden_dim=cfg.planner.bridge_hidden_dim,
    )

    train_dataset = PlannerActionDataset(
        train_examples,
        tokenizer=tokenizer,
        max_prompt_length=cfg.training.max_prompt_length,
        max_completion_length=cfg.training.max_completion_length,
    )
    eval_dataset = PlannerActionDataset(
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
        gradient_checkpointing=False,
        seed=cfg.training.seed,
    )
    trainer = LatentPlannerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=PlannerCollator(tokenizer),
        tokenizer=tokenizer,
    )
    trainer.train()

    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(model.planner_model, "peft_config"):
        model.planner_model.save_pretrained(str(final_dir / "planner_lora"))
        tokenizer.save_pretrained(str(final_dir / "planner_lora"))
    else:
        tokenizer.save_pretrained(str(final_dir / "planner_tokenizer"))
    torch.save(
        {
            "kv_bridge": model.kv_bridge.state_dict(),
            "belief_projector": model.belief_projector.state_dict(),
            "candidate_diseases": cfg.planner.candidate_diseases,
            "memory_tokens_per_layer": cfg.planner.memory_tokens_per_layer,
            "num_belief_tokens": cfg.planner.num_belief_tokens,
        },
        final_dir / "latent_bridges.pt",
    )
    return output_dir
