from pathlib import Path

from data.mimic import DISEASES, MIMICStateBuilder


def build_state_builder(
    builder_name: str,
    dataset_root: str | Path,
    diagnosis_prompt_file: str | Path,
    lab_test_mapping_file: str | Path | None = None,
):
    if builder_name == "mimic_cdm":
        dataset_root = Path(dataset_root)
        return MIMICStateBuilder(
            lab_test_mapping_file=lab_test_mapping_file or dataset_root / "lab_test_mapping.csv",
            diagnosis_prompt_file=diagnosis_prompt_file,
        )
    if builder_name == "rare_disease":
        raise NotImplementedError(
            "The rare_disease builder is not implemented yet. Add a renderer under "
            "src/data/ after the dataset schema is finalized."
        )
    raise ValueError(f"Unsupported data builder: {builder_name}")


__all__ = ["DISEASES", "MIMICStateBuilder", "build_state_builder"]
