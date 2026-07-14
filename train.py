import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from config import load_config
from training.diagnosis_adapter import train_diagnosis_adapter
from training.planner_adapter import train_planner_adapter


def main() -> None:
    cfg = load_config()
    if cfg.train.diagnosis_adapter:
        train_diagnosis_adapter(cfg)
    elif cfg.train.planner_adapter:
        train_planner_adapter(cfg)
    else:
        raise ValueError(
            "No enabled module. Set train.diagnosis_adapter=true or train.planner_adapter=true "
            "in the config."
        )


if __name__ == "__main__":
    main()
