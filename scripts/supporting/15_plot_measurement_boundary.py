#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "results"
    / "tables"
    / "metr_late_common_sample_boundary.csv"
)

OUT_PDF = (
    ROOT
    / "results"
    / "figures"
    / "fig_measurement_boundary_v3.pdf"
)

OUT_PNG = (
    ROOT
    / "results"
    / "figures"
    / "fig_measurement_boundary_v3.png"
)


def main():

    df = pd.read_csv(INPUT)

    order = [
        "All developers",
        "Panel developers",
        "New developers",
    ]

    df = (
        df.set_index("cohort")
        .loc[order]
        .reset_index()
    )

    fig, ax = plt.subplots(
        figsize=(7.2, 4.8)
    )

    x_initial = 0.0
    x_total = 1.0

    # Small horizontal offsets prevent overlapping CIs.
    offsets = {
        "All developers": -0.045,
        "Panel developers": 0.000,
        "New developers": +0.045,
    }

    for _, row in df.iterrows():

        cohort = row["cohort"]

        x1 = (
            x_initial
            + offsets[cohort]
        )

        x2 = (
            x_total
            + offsets[cohort]
        )

        # -----------------------------------------------------
        # Point estimates
        # -----------------------------------------------------

        initial = row[
            "initial_productivity_effect_pct"
        ]

        total = row[
            "total_productivity_effect_pct"
        ]

        # -----------------------------------------------------
        # 95% confidence intervals
        # -----------------------------------------------------

        initial_low = row[
            "initial_productivity_ci_low_pct"
        ]

        initial_high = row[
            "initial_productivity_ci_high_pct"
        ]

        total_low = row[
            "total_productivity_ci_low_pct"
        ]

        total_high = row[
            "total_productivity_ci_high_pct"
        ]

        # -----------------------------------------------------
        # Connector line
        #
        # Capture its automatically assigned matplotlib color,
        # then reuse the SAME color for both point estimates
        # and confidence intervals.
        # -----------------------------------------------------

        line, = ax.plot(
            [x1, x2],
            [initial, total],
            linewidth=1.7,
            alpha=0.82,
            label=cohort,
            zorder=2
        )

        cohort_color = (
            line.get_color()
        )

        # -----------------------------------------------------
        # Initial implementation estimate
        # -----------------------------------------------------

        initial_yerr = np.array(
            [
                [
                    initial
                    - initial_low
                ],
                [
                    initial_high
                    - initial
                ],
            ]
        )

        ax.errorbar(
            x1,
            initial,
            yerr=initial_yerr,
            fmt="o",
            capsize=4,
            capthick=1.2,
            elinewidth=1.3,
            markersize=6.5,
            color=cohort_color,
            zorder=3
        )

        # -----------------------------------------------------
        # Initial + post-review estimate
        # -----------------------------------------------------

        total_yerr = np.array(
            [
                [
                    total
                    - total_low
                ],
                [
                    total_high
                    - total
                ],
            ]
        )

        ax.errorbar(
            x2,
            total,
            yerr=total_yerr,
            fmt="o",
            capsize=4,
            capthick=1.2,
            elinewidth=1.3,
            markersize=6.5,
            color=cohort_color,
            zorder=3
        )

    # ---------------------------------------------------------
    # Zero-reference line
    # ---------------------------------------------------------

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1.0,
        alpha=0.65,
        zorder=1
    )

    # ---------------------------------------------------------
    # Axes
    # ---------------------------------------------------------

    ax.set_xticks(
        [
            x_initial,
            x_total,
        ]
    )

    ax.set_xticklabels(
        [
            "Initial\nimplementation",
            "Initial +\npost-review",
        ]
    )

    ax.set_ylabel(
        "Estimated AI productivity effect (%)"
    )

    # Slight horizontal padding so error bars and points
    # do not sit too close to the plot boundary.
    ax.set_xlim(
        -0.12,
        1.12
    )

    # ---------------------------------------------------------
    # Grid
    # ---------------------------------------------------------

    ax.grid(
        axis="y",
        linestyle=":",
        alpha=0.28,
        zorder=0
    )

    # ---------------------------------------------------------
    # Legend
    # ---------------------------------------------------------

    ax.legend(
        frameon=False,
        loc="upper right",
        fontsize=10
    )

    # ---------------------------------------------------------
    # Layout + export
    # ---------------------------------------------------------

    fig.tight_layout()

    OUT_PDF.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fig.savefig(
        OUT_PDF,
        bbox_inches="tight"
    )

    fig.savefig(
        OUT_PNG,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"PDF written to: {OUT_PDF}"
    )

    print(
        f"PNG written to: {OUT_PNG}"
    )


if __name__ == "__main__":
    main()
###########################
# #!/usr/bin/env python3

# from pathlib import Path

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt


# ROOT = Path(__file__).resolve().parents[1]

# INPUT = (
#     ROOT
#     / "results"
#     / "tables"
#     / "metr_late_common_sample_boundary.csv"
# )

# OUT_PDF = (
#     ROOT
#     / "results"
#     / "figures"
#     / "fig_measurement_boundary_v2.pdf"
# )

# OUT_PNG = (
#     ROOT
#     / "results"
#     / "figures"
#     / "fig_measurement_boundary_v2.png"
# )


# def main():

#     df = pd.read_csv(INPUT)

#     order = [
#         "All developers",
#         "Panel developers",
#         "New developers",
#     ]

#     df = (
#         df.set_index("cohort")
#         .loc[order]
#         .reset_index()
#     )

#     fig, ax = plt.subplots(
#         figsize=(7.2, 4.8)
#     )

#     x_initial = 0.0
#     x_total = 1.0

#     offsets = {
#         "All developers": -0.04,
#         "Panel developers": 0.00,
#         "New developers": +0.04,
#     }

#     for _, row in df.iterrows():

#         cohort = row["cohort"]

#         x1 = (
#             x_initial
#             + offsets[cohort]
#         )

#         x2 = (
#             x_total
#             + offsets[cohort]
#         )

#         initial = (
#             row[
#                 "initial_productivity_effect_pct"
#             ]
#         )

#         total = (
#             row[
#                 "total_productivity_effect_pct"
#             ]
#         )

#         initial_low = (
#             row[
#                 "initial_productivity_ci_low_pct"
#             ]
#         )

#         initial_high = (
#             row[
#                 "initial_productivity_ci_high_pct"
#             ]
#         )

#         total_low = (
#             row[
#                 "total_productivity_ci_low_pct"
#             ]
#         )

#         total_high = (
#             row[
#                 "total_productivity_ci_high_pct"
#             ]
#         )

#         # -----------------------------------------------------
#         # Connector
#         # -----------------------------------------------------

#         ax.plot(
#             [x1, x2],
#             [initial, total],
#             linewidth=1.6,
#             alpha=0.75,
#             label=cohort
#         )

#         # -----------------------------------------------------
#         # Initial point + CI
#         # -----------------------------------------------------

#         ax.errorbar(
#             x1,
#             initial,
#             yerr=np.array(
#                 [
#                     [
#                         initial
#                         - initial_low
#                     ],
#                     [
#                         initial_high
#                         - initial
#                     ],
#                 ]
#             ),
#             fmt="o",
#             capsize=4,
#             elinewidth=1.3,
#             markersize=6
#         )

#         # -----------------------------------------------------
#         # Total point + CI
#         # -----------------------------------------------------

#         ax.errorbar(
#             x2,
#             total,
#             yerr=np.array(
#                 [
#                     [
#                         total
#                         - total_low
#                     ],
#                     [
#                         total_high
#                         - total
#                     ],
#                 ]
#             ),
#             fmt="o",
#             capsize=4,
#             elinewidth=1.3,
#             markersize=6
#         )

#     # ---------------------------------------------------------
#     # Reference line
#     # ---------------------------------------------------------

#     ax.axhline(
#         0,
#         linestyle="--",
#         linewidth=1,
#         alpha=0.65
#     )

#     # ---------------------------------------------------------
#     # Axes
#     # ---------------------------------------------------------

#     ax.set_xticks(
#         [
#             x_initial,
#             x_total,
#         ]
#     )

#     ax.set_xticklabels(
#         [
#             "Initial\nimplementation",
#             "Initial +\npost-review",
#         ]
#     )

#     ax.set_ylabel(
#         "Estimated AI productivity effect (%)"
#     )

#     ax.grid(
#         axis="y",
#         linestyle=":",
#         alpha=0.28
#     )

#     # ---------------------------------------------------------
#     # Legend
#     # ---------------------------------------------------------

#     ax.legend(
#         frameon=False,
#         loc="upper right"
#     )

#     fig.tight_layout()

#     OUT_PDF.parent.mkdir(
#         parents=True,
#         exist_ok=True
#     )

#     fig.savefig(
#         OUT_PDF,
#         bbox_inches="tight"
#     )

#     fig.savefig(
#         OUT_PNG,
#         dpi=300,
#         bbox_inches="tight"
#     )

#     plt.close(fig)

#     print(
#         f"PDF written to: {OUT_PDF}"
#     )

#     print(
#         f"PNG written to: {OUT_PNG}"
#     )


# if __name__ == "__main__":
#     main()
# ###############
# # #!/usr/bin/env python3

# # from pathlib import Path

# # import pandas as pd
# # import matplotlib.pyplot as plt


# # ROOT = Path(__file__).resolve().parents[1]

# # INPUT = (
# #     ROOT
# #     / "results"
# #     / "tables"
# #     / "metr_late_common_sample_boundary.csv"
# # )

# # OUT_PDF = (
# #     ROOT
# #     / "results"
# #     / "figures"
# #     / "fig_measurement_boundary.pdf"
# # )

# # OUT_PNG = (
# #     ROOT
# #     / "results"
# #     / "figures"
# #     / "fig_measurement_boundary.png"
# # )


# # def main():
# #     df = pd.read_csv(INPUT)

# #     order = [
# #         "All developers",
# #         "Panel developers",
# #         "New developers",
# #     ]

# #     df = (
# #         df.set_index("cohort")
# #         .loc[order]
# #         .reset_index()
# #     )

# #     fig, ax = plt.subplots(
# #         figsize=(7.0, 4.7)
# #     )

# #     x_initial = 0
# #     x_total = 1

# #     for _, row in df.iterrows():

# #         initial = (
# #             row[
# #                 "initial_productivity_effect_pct"
# #             ]
# #         )

# #         total = (
# #             row[
# #                 "total_productivity_effect_pct"
# #             ]
# #         )

# #         ax.plot(
# #             [x_initial, x_total],
# #             [initial, total],
# #             marker="o",
# #             linewidth=1.8,
# #             label=row["cohort"]
# #         )

# #     ax.axhline(
# #         0,
# #         linestyle="--",
# #         linewidth=1,
# #         alpha=0.7
# #     )

# #     ax.set_xticks(
# #         [x_initial, x_total]
# #     )

# #     ax.set_xticklabels(
# #         [
# #             "Initial\nimplementation",
# #             "Initial +\npost-review",
# #         ]
# #     )

# #     ax.set_ylabel(
# #         "Estimated AI productivity effect (%)"
# #     )

# #     ax.grid(
# #         axis="y",
# #         linestyle=":",
# #         alpha=0.35
# #     )

# #     ax.legend(
# #         frameon=False
# #     )

# #     fig.tight_layout()

# #     OUT_PDF.parent.mkdir(
# #         parents=True,
# #         exist_ok=True
# #     )

# #     fig.savefig(
# #         OUT_PDF,
# #         bbox_inches="tight"
# #     )

# #     fig.savefig(
# #         OUT_PNG,
# #         dpi=300,
# #         bbox_inches="tight"
# #     )

# #     plt.close(fig)

# #     print(
# #         f"PDF written to: {OUT_PDF}"
# #     )

# #     print(
# #         f"PNG written to: {OUT_PNG}"
# #     )


# # if __name__ == "__main__":
# #     main()
