import ast
import random
from pathlib import Path
from typing import Any

import pandas as pd

from data.text import extract_findings_from_report


LAB_TESTS = [
    "Complete Blood Count",
    "Basic Metabolic Panel",
    "Comprehensive Metabolic Panel",
    "Renal Function Panel",
    "Liver Function Panel",
    "Urinalysis",
    "Electrolyte Panel",
]
IMAGING_TESTS = ["CT", "MRI", "Radiograph", "Ultrasound"]
OTHER_TESTS = ["Physical Examination"]
DISEASES = ["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis"]


def load_prompt_template(prompt_file: str | Path) -> str:
    template = Path(prompt_file).read_text(encoding="utf-8")
    if "{patient_state}" not in template:
        raise ValueError(f"Prompt template must contain {{patient_state}}: {prompt_file}")
    return template


class MIMICStateBuilder:
    """Render MIMIC-CDM rows into partial patient states for diagnosis-only SFT."""

    def __init__(
        self,
        lab_test_mapping_file: str | Path,
        diagnosis_prompt_file: str | Path,
        lab_tests: list[str] | None = None,
        imaging_tests: list[str] | None = None,
        other_tests: list[str] | None = None,
    ) -> None:
        self.lab_tests = lab_tests or LAB_TESTS
        self.imaging_tests = imaging_tests or IMAGING_TESTS
        self.other_tests = other_tests or OTHER_TESTS
        self.all_tests = self.lab_tests + self.imaging_tests + self.other_tests
        self.prompt_template = load_prompt_template(diagnosis_prompt_file)
        self.lab_mapping = pd.read_csv(lab_test_mapping_file)
        self.test_ids: dict[str, list[int]] = {}
        self.id_to_name: dict[int, str] = {}
        self._index_lab_mapping()

    def _index_lab_mapping(self) -> None:
        for test in self.lab_tests:
            row = self.lab_mapping[self.lab_mapping["label"] == test]
            if row.empty:
                self.test_ids[test] = []
                continue
            ids = ast.literal_eval(row["corresponding_ids"].item())
            self.test_ids[test] = ids
            for item_id in ids:
                mapped = self.lab_mapping[self.lab_mapping["itemid"] == item_id]
                if mapped.empty:
                    self.id_to_name[item_id] = str(item_id)
                    continue
                corr = ast.literal_eval(mapped["corresponding_ids"].item())
                canonical = self.lab_mapping[self.lab_mapping["itemid"] == corr[0]]
                self.id_to_name[item_id] = (
                    canonical["label"].item() if not canonical.empty else mapped["label"].item()
                )

    def build_examples(
        self,
        data_file: str | Path,
        seed: int,
        max_state_chars: int,
        min_visible_tests: int = 1,
        max_visible_tests: int | None = None,
        states_per_patient: int = 1,
        limit: int | None = None,
        use_all_available_tests: bool = False,
    ) -> list[dict[str, Any]]:
        df = pd.read_csv(data_file)
        if limit:
            df = df.iloc[:limit].reset_index(drop=True)

        max_visible_tests = max_visible_tests or len(self.all_tests)
        min_visible_tests = max(1, min_visible_tests)
        max_visible_tests = min(len(self.all_tests), max_visible_tests)
        if min_visible_tests > max_visible_tests:
            raise ValueError("min_visible_tests must be <= max_visible_tests")

        examples = []
        for row_idx, row in df.iterrows():
            available_tests = self.available_tests(row)
            for state_idx in range(states_per_patient):
                rng = random.Random(seed + int(row_idx) * 1009 + state_idx)
                if use_all_available_tests:
                    visible_tests = list(available_tests)
                else:
                    visible_tests = self.sample_visible_tests(
                        available_tests=available_tests,
                        rng=rng,
                        min_visible_tests=min_visible_tests,
                        max_visible_tests=max_visible_tests,
                    )
                examples.append(
                    self.build_example(
                        row=row,
                        visible_tests=visible_tests,
                        max_state_chars=max_state_chars,
                        state_sample_id=state_idx,
                    )
                )
        return examples

    def build_example(
        self,
        row: pd.Series,
        visible_tests: list[str],
        max_state_chars: int,
        state_sample_id: int,
    ) -> dict[str, Any]:
        rendered_tests = self.render_available_tests(row, visible_tests)
        state = self.render_state(row, rendered_tests)
        if len(state) > max_state_chars:
            state = state[-max_state_chars:]
        label = str(row["Label"]).strip().lower()
        included_tests = [test for test, _ in rendered_tests]
        return {
            "prompt": self.render_prompt(state),
            "target": label,
            "condition": label,
            "patient_id": row["Patient ID"],
            "visible_tests": ", ".join(included_tests),
            "visible_test_count": len(included_tests),
            "state_sample_id": state_sample_id,
        }

    def render_prompt(self, state: str) -> str:
        return self.prompt_template.format(patient_state=state)

    def render_state(self, row: pd.Series, rendered_tests: list[tuple[str, str]]) -> str:
        patient_history = row.get("Patient History Summary", "")
        state = f"Patient History: {patient_history}\n\n"
        state += "\n\n".join(f"{test}: {result}" for test, result in rendered_tests)
        return state

    def render_available_tests(self, row: pd.Series, visible_tests: list[str]) -> list[tuple[str, str]]:
        rendered_tests = []
        for test in visible_tests:
            result = self.render_test(row, test)
            if result.strip().lower() == "not available.":
                continue
            rendered_tests.append((test, result))
        return rendered_tests

    def available_tests(self, row: pd.Series) -> list[str]:
        return [
            test
            for test in self.all_tests
            if self.render_test(row, test).strip().lower() != "not available."
        ]

    def sample_visible_tests(
        self,
        available_tests: list[str],
        rng: random.Random,
        min_visible_tests: int,
        max_visible_tests: int,
    ) -> list[str]:
        if not available_tests:
            return []
        upper = min(max_visible_tests, len(available_tests))
        lower = min(min_visible_tests, upper)
        test_count = rng.randint(lower, upper)
        return rng.sample(available_tests, test_count)

    def render_test(self, row: pd.Series, test: str) -> str:
        if test == "Physical Examination":
            physical_exam = row.get("Physical Examination", None)
            if physical_exam is None or pd.isna(physical_exam) or str(physical_exam).strip() == "":
                return "not available.\n"
            return str(physical_exam)
        if test in self.lab_tests:
            return self.render_lab_test(row, test)
        if test in self.imaging_tests:
            return self.render_imaging_test(row, test)
        return "not available.\n"

    def render_lab_test(self, row: pd.Series, test: str) -> str:
        try:
            lab_results = ast.literal_eval(row.get("Laboratory Tests", "{}"))
        except (ValueError, SyntaxError):
            lab_results = {}

        parts = []
        for item_id in self.test_ids.get(test, []):
            if item_id in lab_results:
                parts.append(f"{self.id_to_name.get(item_id, item_id)}: {lab_results[item_id]}")
        return ", ".join(parts) + "\n" if parts else "not available.\n"

    def render_imaging_test(self, row: pd.Series, test: str) -> str:
        try:
            radiology = ast.literal_eval(row.get("Radiology", "[]"))
        except (ValueError, SyntaxError):
            radiology = []

        reports = [
            extract_findings_from_report(result.get("Report"))
            for result in radiology
            if isinstance(result, dict) and result.get("Modality") == test
        ]
        if not reports:
            return "not available.\n"
        if len(reports) == 1:
            return reports[0]

        report_string = ""
        for idx, report in enumerate(reports):
            report_string += report
            if idx != len(reports) - 1:
                report_string += f"\nReport {idx + 2}:\n"
        return f"There are {len(reports)} reports available for this test:\n{report_string}"
