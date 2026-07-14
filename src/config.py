import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrainConfig:
    diagnosis_adapter: bool = True
    planner_adapter: bool = False


@dataclass
class ModelConfig:
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    quant_4bit: bool = False
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


@dataclass
class DataConfig:
    name: str = "mimic_cdm"
    builder: str = "mimic_cdm"
    root_dir: str = "dataset"
    train_file: str = "train.csv"
    val_file: str = "val.csv"
    test_file: str = "test.csv"
    lab_test_mapping_file: str = "lab_test_mapping.csv"
    diagnosis_prompt_file: str = "prompts/diagnosis_adapter.txt"
    max_state_chars: int = 12000
    smoke_limit: int | None = None


@dataclass
class TrainingConfig:
    run_root: str = "/media/NAS/nas_175/seojun/LatentCDM/runs"
    experiment_name: str = "diagnosis_adapter"
    output_dir: str = ""
    timestamp_output_dir: bool = True
    seed: int = 269
    bf16: bool = True
    report_to: str = "none"
    num_train_epochs: float = 4.0
    learning_rate: float = 1e-5
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = True
    logging_steps: int = 5
    save_steps: int = 100
    min_visible_tests: int = 1
    max_visible_tests: int = 12
    states_per_patient: int = 4
    use_all_available_tests: bool = False
    max_prompt_length: int = 4096
    max_completion_length: int = 32


@dataclass
class PlannerConfig:
    diagnosis_adapter_dir: str = ""
    memory_tokens_per_layer: int = 1
    num_belief_tokens: int = 1
    bridge_hidden_dim: int = 2048
    candidate_diseases: list[str] = field(
        default_factory=lambda: [
            "appendicitis",
            "cholecystitis",
            "diverticulitis",
            "pancreatitis",
        ]
    )
    action_prefix: str = "Action:"


@dataclass
class ExperimentConfig:
    root_dir: Path
    config_path: Path
    train: TrainConfig = field(default_factory=TrainConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)


def _coerce_value(current: Any, value: Any) -> Any:
    if value is None:
        return None
    if isinstance(current, bool) and not isinstance(value, bool):
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    if isinstance(current, int) and not isinstance(current, bool) and not isinstance(value, int):
        return int(value)
    if isinstance(current, float) and not isinstance(value, float):
        return float(value)
    return value


def _merge_dataclass(obj: Any, values: dict[str, Any]) -> Any:
    for key, value in values.items():
        current = getattr(obj, key)
        if hasattr(current, "__dataclass_fields__"):
            _merge_dataclass(current, value)
        else:
            setattr(obj, key, _coerce_value(current, value))
    return obj


def resolve_path(root_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root_dir / path).resolve()


def resolve_output_base(cfg: ExperimentConfig) -> Path:
    if cfg.training.output_dir:
        return resolve_path(cfg.root_dir, cfg.training.output_dir)
    return (Path(cfg.training.run_root) / cfg.training.experiment_name / cfg.data.name).resolve()


def resolve_dataset_path(cfg: ExperimentConfig, filename: str | Path) -> Path:
    path = Path(filename)
    if path.is_absolute():
        return path
    return (resolve_path(cfg.root_dir, cfg.data.root_dir) / path).resolve()


def load_config() -> ExperimentConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    root_dir = config_path.parent.parent.resolve()
    cfg = ExperimentConfig(root_dir=root_dir, config_path=config_path)

    with config_path.open("r", encoding="utf-8") as f:
        values = yaml.safe_load(f) or {}
    return _merge_dataclass(cfg, values)
