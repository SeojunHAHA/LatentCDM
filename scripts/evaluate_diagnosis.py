#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_config, resolve_dataset_path, resolve_path  # noqa: E402
from data import build_state_builder  # noqa: E402
from data.mimic import DISEASES  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--use-all-available-tests", action="store_true")
    return parser.parse_args()


def normalize_prediction(text: str) -> str:
    normalized = text.strip().lower()
    normalized = normalized.splitlines()[0].strip() if normalized else ""
    normalized = normalized.strip(" .,:;`'\"")
    for disease in DISEASES:
        if normalized == disease:
            return disease
    for disease in DISEASES:
        if disease in normalized:
            return disease
    return normalized


def batch_iter(items: list[dict[str, Any]], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def main() -> None:
    args = parse_args()

    sys.argv = [sys.argv[0], "--config", args.config]
    cfg = load_config()

    split_file = {
        "train": cfg.data.train_file,
        "val": cfg.data.val_file,
        "test": cfg.data.test_file,
    }[args.split]

    dataset_root = resolve_path(cfg.root_dir, cfg.data.root_dir)
    builder = build_state_builder(
        builder_name=cfg.data.builder,
        dataset_root=dataset_root,
        diagnosis_prompt_file=resolve_path(cfg.root_dir, cfg.data.diagnosis_prompt_file),
        lab_test_mapping_file=resolve_dataset_path(cfg, cfg.data.lab_test_mapping_file),
    )
    examples = builder.build_examples(
        data_file=resolve_dataset_path(cfg, split_file),
        seed=cfg.training.seed + 101,
        max_state_chars=cfg.data.max_state_chars,
        min_visible_tests=cfg.training.min_visible_tests,
        max_visible_tests=cfg.training.max_visible_tests,
        states_per_patient=1,
        limit=args.max_samples,
        use_all_available_tests=args.use_all_available_tests,
    )

    adapter_dir = Path(args.adapter_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else adapter_dir.parent / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    rows = []
    correct = 0
    with torch.no_grad():
        for batch in batch_iter(examples, args.batch_size):
            encoded = tokenizer(
                [item["prompt"] for item in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=cfg.training.max_prompt_length,
                add_special_tokens=False,
            ).to(model.device)
            outputs = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            prompt_len = encoded["input_ids"].shape[1]
            for item, output_ids in zip(batch, outputs):
                generated_ids = output_ids[prompt_len:]
                raw_prediction = tokenizer.decode(generated_ids, skip_special_tokens=True)
                prediction = normalize_prediction(raw_prediction)
                target = str(item["target"]).strip().lower()
                is_correct = prediction == target
                correct += int(is_correct)
                rows.append(
                    {
                        "patient_id": item["patient_id"],
                        "target": target,
                        "prediction": prediction,
                        "raw_prediction": raw_prediction.strip(),
                        "correct": is_correct,
                        "visible_tests": item["visible_tests"],
                        "visible_test_count": item["visible_test_count"],
                    }
                )

    total = len(rows)
    metrics = {
        "adapter_dir": str(adapter_dir),
        "config": str(Path(args.config).resolve()),
        "split": args.split,
        "examples": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "use_all_available_tests": args.use_all_available_tests,
        "max_samples": args.max_samples,
    }

    (output_dir / f"{args.split}_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    with (output_dir / f"{args.split}_predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
