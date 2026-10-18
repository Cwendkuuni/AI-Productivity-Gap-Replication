#!/usr/bin/env python3

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

XLSX = (
    ROOT
    / "data"
    / "raw"
    / "slr_artifact"
    / "SLR_all_included_studies.xlsx"
)

OUT = (
    ROOT
    / "results"
    / "tables"
    / "slr_space_mapping.csv"
)

SUMMARY_OUT = (
    ROOT
    / "results"
    / "tables"
    / "slr_space_summary.csv"
)


# =============================================================
# Mapping transcribed directly from Table 2 of Tables.pdf.
#
# IMPORTANT:
# The table currently covers PS1--PS37 only.
# PS38 and PS39 are intentionally NOT inferred.
# =============================================================

SPACE_MAPPING = {

    7:  ["Satisfaction"],
    3:  ["Performance"],
    1:  ["Communication"],

    10: ["Activity", "Efficiency"],

    30: ["Performance", "Activity"],

    2:  ["Performance", "Efficiency"],
    12: ["Performance", "Efficiency"],
    25: ["Performance", "Efficiency"],

    6:  ["Satisfaction", "Communication"],
    8:  ["Satisfaction", "Communication"],
    11: ["Satisfaction", "Communication"],

    15: ["Satisfaction", "Efficiency"],
    17: ["Satisfaction", "Efficiency"],
    22: ["Satisfaction", "Efficiency"],
    26: ["Satisfaction", "Efficiency"],

    9:  ["Satisfaction", "Performance"],
    13: ["Satisfaction", "Performance"],
    32: ["Satisfaction", "Performance"],
    35: ["Satisfaction", "Performance"],

    29: [
        "Performance",
        "Communication",
        "Efficiency",
    ],

    27: [
        "Satisfaction",
        "Activity",
        "Efficiency",
    ],

    24: [
        "Satisfaction",
        "Performance",
        "Communication",
    ],

    21: [
        "Satisfaction",
        "Communication",
        "Efficiency",
    ],

    31: [
        "Satisfaction",
        "Communication",
        "Efficiency",
    ],

    5: [
        "Satisfaction",
        "Performance",
        "Activity",
    ],

    18: [
        "Satisfaction",
        "Performance",
        "Activity",
    ],

    20: [
        "Satisfaction",
        "Performance",
        "Activity",
    ],

    4: [
        "Satisfaction",
        "Performance",
        "Efficiency",
    ],

    16: [
        "Satisfaction",
        "Performance",
        "Efficiency",
    ],

    23: [
        "Satisfaction",
        "Performance",
        "Efficiency",
    ],

    33: [
        "Satisfaction",
        "Performance",
        "Efficiency",
    ],

    36: [
        "Satisfaction",
        "Performance",
        "Efficiency",
    ],

    14: [
        "Satisfaction",
        "Performance",
        "Communication",
        "Efficiency",
    ],

    19: [
        "Satisfaction",
        "Performance",
        "Activity",
        "Communication",
        "Efficiency",
    ],

    28: [
        "Satisfaction",
        "Performance",
        "Activity",
        "Communication",
        "Efficiency",
    ],

    34: [
        "Satisfaction",
        "Performance",
        "Activity",
        "Communication",
        "Efficiency",
    ],

    37: [
        "Satisfaction",
        "Performance",
        "Activity",
        "Communication",
        "Efficiency",
    ],
}


DIMENSIONS = [
    "Satisfaction",
    "Performance",
    "Activity",
    "Communication",
    "Efficiency",
]


def main():

    df = pd.read_excel(
        XLSX
    )

    # ---------------------------------------------------------
    # Verify study index
    # ---------------------------------------------------------

    df["count"] = pd.to_numeric(
        df["count"],
        errors="raise"
    ).astype(int)

    df["PS"] = (
        "PS"
        + df["count"].astype(str)
    )

    # ---------------------------------------------------------
    # Add SPACE classification
    # ---------------------------------------------------------

    for dimension in DIMENSIONS:

        df[dimension] = df["count"].apply(
            lambda ps: (
                dimension
                in SPACE_MAPPING.get(
                    ps,
                    []
                )
            )
        )

    df["space_dimension_count"] = (
        df[DIMENSIONS]
        .sum(axis=1)
    )

    df["space_dimensions"] = (
        df["count"]
        .apply(
            lambda ps: (
                " + ".join(
                    SPACE_MAPPING[ps]
                )
                if ps in SPACE_MAPPING
                else ""
            )
        )
    )

    df["space_mapping_status"] = (
        df["count"]
        .apply(
            lambda ps: (
                "Mapped from Tables.pdf"
                if ps in SPACE_MAPPING
                else "Not present in Tables.pdf"
            )
        )
    )

    # ---------------------------------------------------------
    # Save study-level mapping
    # ---------------------------------------------------------

    keep_cols = [
        "count",
        "PS",
        "Title",
        "Strategy",
        "Procedure",
        "Formative vs Summative",
        "Qualtative, Quantitative, Both",
        "Satisfaction",
        "Performance",
        "Activity",
        "Communication",
        "Efficiency",
        "space_dimension_count",
        "space_dimensions",
        "space_mapping_status",
    ]

    df[keep_cols].to_csv(
        OUT,
        index=False
    )

    # ---------------------------------------------------------
    # Summary ONLY over studies explicitly present
    # in Tables.pdf.
    # ---------------------------------------------------------

    mapped = df[
        df["count"].isin(
            SPACE_MAPPING
        )
    ].copy()

    summary_rows = []

    for dimension in DIMENSIONS:

        n = int(
            mapped[dimension].sum()
        )

        pct = (
            100.0
            * n
            / len(mapped)
        )

        summary_rows.append(
            {
                "dimension": dimension,
                "studies": n,
                "denominator": len(mapped),
                "percent": round(
                    pct,
                    2
                ),
            }
        )

    # Coverage breadth
    for k in range(
        1,
        6
    ):

        n = int(
            (
                mapped[
                    "space_dimension_count"
                ]
                == k
            ).sum()
        )

        summary_rows.append(
            {
                "dimension": (
                    f"Exactly {k} SPACE dimensions"
                ),
                "studies": n,
                "denominator": len(mapped),
                "percent": round(
                    100.0
                    * n
                    / len(mapped),
                    2
                ),
            }
        )

    n_all = int(
        (
            mapped[
                "space_dimension_count"
            ]
            == 5
        ).sum()
    )

    summary_rows.append(
        {
            "dimension": "All five SPACE dimensions",
            "studies": n_all,
            "denominator": len(mapped),
            "percent": round(
                100.0
                * n_all
                / len(mapped),
                2
            ),
        }
    )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        SUMMARY_OUT,
        index=False
    )

    # ---------------------------------------------------------
    # Console report
    # ---------------------------------------------------------

    print("=" * 72)
    print("SLR SPACE RECONCILIATION")
    print("=" * 72)

    print(
        f"Studies in XLSX:             {len(df)}"
    )

    print(
        f"Studies mapped in Table 2:   {len(mapped)}"
    )

    missing = df.loc[
        ~df["count"].isin(
            SPACE_MAPPING
        ),
        [
            "PS",
            "Title",
        ]
    ]

    print()
    print(
        "Studies absent from Table 2:"
    )

    print(
        missing.to_string(
            index=False
        )
    )

    print()
    print(
        "SPACE dimension prevalence "
        "(denominator = mapped studies only)"
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

    print()

    print(
        f"Study-level mapping: {OUT}"
    )

    print(
        f"Summary:             {SUMMARY_OUT}"
    )


if __name__ == "__main__":
    main()
