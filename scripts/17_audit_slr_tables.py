#!/usr/bin/env python3

from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]

PDF = (
    ROOT
    / "data"
    / "raw"
    / "slr_artifact"
    / "Tables.pdf"
)

OUT_DIR = ROOT / "results" / "logs"
OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TEXT_FILE = (
    OUT_DIR
    / "slr_tables.txt"
)

AUDIT_FILE = (
    OUT_DIR
    / "slr_tables_audit.txt"
)


KEYWORD_GROUPS = {
    "productivity": [
        "productivity",
        "productive",
    ],

    "measurement": [
        "metric",
        "metrics",
        "measure",
        "measurement",
        "outcome",
    ],

    "SPACE": [
        "space",
        "satisfaction",
        "performance",
        "activity",
        "communication",
        "collaboration",
        "efficiency",
    ],

    "effort_time": [
        "effort",
        "time",
        "duration",
        "completion time",
        "task time",
    ],

    "quality": [
        "quality",
        "correctness",
        "accuracy",
        "defect",
        "bug",
    ],

    "review_verification": [
        "review",
        "verification",
        "validate",
        "validation",
        "inspection",
    ],

    "testing_rework": [
        "test",
        "testing",
        "debug",
        "debugging",
        "rework",
        "repair",
    ],

    "delivery_integration": [
        "integration",
        "delivery",
        "deployment",
        "merge",
        "pull request",
        "ci",
    ],

    "subjective": [
        "survey",
        "self-report",
        "self reported",
        "perceived",
        "perception",
        "interview",
    ],

    "tool_interaction": [
        "acceptance",
        "suggestion",
        "prompt",
        "interaction",
        "generated code",
        "lines of code",
    ],
}


def extract_text():

    cmd = [
        "pdftotext",
        "-layout",
        str(PDF),
        str(TEXT_FILE),
    ]

    subprocess.run(
        cmd,
        check=True
    )


def context(lines, index, radius=2):

    start = max(
        0,
        index - radius
    )

    end = min(
        len(lines),
        index + radius + 1
    )

    return "\n".join(
        lines[start:end]
    )


def main():

    if not PDF.exists():
        raise FileNotFoundError(
            PDF
        )

    extract_text()

    text = TEXT_FILE.read_text(
        encoding="utf-8",
        errors="replace"
    )

    lines = text.splitlines()

    out = []

    out.append(
        "=" * 80
    )

    out.append(
        "SLR TABLES.PDF AUDIT"
    )

    out.append(
        "=" * 80
    )

    out.append(
        f"PDF: {PDF}"
    )

    out.append(
        f"Extracted lines: {len(lines)}"
    )

    out.append("")

    # ---------------------------------------------------------
    # Detect likely table / RQ headings
    # ---------------------------------------------------------

    heading_pattern = re.compile(
        r"\b("
        r"table|rq1|rq2|rq3|"
        r"research question|dimension|"
        r"metric|measure"
        r")\b",
        re.IGNORECASE
    )

    out.append(
        "=" * 80
    )

    out.append(
        "LIKELY HEADINGS / TABLE MARKERS"
    )

    out.append(
        "=" * 80
    )

    for i, line in enumerate(
        lines
    ):

        if heading_pattern.search(
            line
        ):

            out.append(
                f"[line {i + 1}] "
                f"{line}"
            )

    out.append("")

    # ---------------------------------------------------------
    # Keyword groups
    # ---------------------------------------------------------

    for group, keywords in (
        KEYWORD_GROUPS.items()
    ):

        out.append(
            "=" * 80
        )

        out.append(
            f"GROUP: {group}"
        )

        out.append(
            "=" * 80
        )

        matched_indexes = []

        for i, line in enumerate(
            lines
        ):

            lower = line.lower()

            if any(
                keyword in lower
                for keyword in keywords
            ):

                matched_indexes.append(
                    i
                )

        out.append(
            f"Matching lines: "
            f"{len(matched_indexes)}"
        )

        out.append("")

        for i in matched_indexes[:50]:

            out.append(
                f"--- line {i + 1} ---"
            )

            out.append(
                context(
                    lines,
                    i,
                    radius=2
                )
            )

            out.append("")

    AUDIT_FILE.write_text(
        "\n".join(out),
        encoding="utf-8"
    )

    print(
        f"Extracted text: {TEXT_FILE}"
    )

    print(
        f"Audit:          {AUDIT_FILE}"
    )


if __name__ == "__main__":
    main()
