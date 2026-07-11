import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from config import load_config
from training.diagnosis_adapter import train_diagnosis_adapter


def main() -> None:
    cfg = load_config()
    if cfg.train.diagnosis_adapter:
        train_diagnosis_adapter(cfg)
    else:
        raise ValueError("No enabled module. Set train.diagnosis_adapter=true in the config.")


if __name__ == "__main__":
    main()
