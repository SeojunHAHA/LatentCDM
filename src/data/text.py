import re


def extract_findings_from_report(input_string: str | None) -> str:
    if not input_string:
        return ""

    sections = re.split(r"(^[A-Z ,_.&]+:\n|(?<=\n)[A-Z ,_.&]+:\n)", input_string)
    section_dict: dict[str, str] = {}
    for idx in range(1, len(sections), 2):
        title = sections[idx].strip()
        content = sections[idx + 1] if idx + 1 < len(sections) else ""
        section_dict[title] = content.strip()

    for removable in ["TECHNIQUE:", "DOSE:", "DLP:"]:
        section_dict.pop(removable, None)

    if "FINDINGS:" in section_dict:
        findings_started = False
        filtered = {}
        for title, content in section_dict.items():
            if title == "FINDINGS:":
                findings_started = True
            if findings_started:
                filtered[title] = content
        section_dict = filtered

    output = ""
    for title, content in section_dict.items():
        output += f"{title}\n{content}\n\n"
    return output.strip()

