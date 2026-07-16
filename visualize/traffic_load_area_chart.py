import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
import numpy as np
import scienceplots
import os
from data import WEIGHT_MLDRL, WEIGHT_SLDRL, algo_colors
import pandas as pd

mpl.rcParams["hatch.linewidth"] = 0.5
plt.style.use(["science", "ieee"])

# Representative (knee) points on the Pareto front, matching visualize/pareto.py
KNEE_W1 = {"MLDRL": 0.55, "SLDRL": 0.30}


def curly_brace(ax, y0, y1, x, width=0.08, lw=0.6, color="black", zorder=5):
    """Draw a vertical curly brace spanning [y0, y1] at horizontal position x.
    The tip pokes out toward x + width (use a negative width to mirror it)."""
    if y1 <= y0:
        return
    ym = (y0 + y1) / 2.0
    q1 = y0 + (ym - y0) * 0.5
    q2 = y1 - (y1 - ym) * 0.5
    verts = [
        (x, y0),
        (x, y0 + (q1 - y0) * 0.5),
        (x + width * 0.5, q1 - (q1 - y0) * 0.5),
        (x + width * 0.5, q1),
        (x + width * 0.5, q1 + (ym - q1) * 0.5),
        (x + width, ym - (ym - q1) * 0.5),
        (x + width, ym),
        (x + width, ym + (q2 - ym) * 0.5),
        (x + width * 0.5, q2 - (q2 - ym) * 0.5),
        (x + width * 0.5, q2),
        (x + width * 0.5, q2 + (y1 - q2) * 0.5),
        (x, y1 - (y1 - q2) * 0.5),
        (x, y1),
    ]
    codes = [Path.MOVETO] + [Path.CURVE4] * 12
    patch = mpatches.PathPatch(
        Path(verts, codes),
        facecolor="none",
        edgecolor=color,
        linewidth=lw,
        zorder=zorder,
        capstyle="round",
    )
    ax.add_patch(patch)


if __name__ == "__main__":
    # Save dir
    SAVE_DIR = "./.output/figures/"
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    labels = ["V2B", "V2V", "V2R-PC5", "V2R-WiFi"]
    colors = [
        "#D6604D",  # 1: V2B
        "#E69F00",  # 2: V2V
        "#4393C3",  # 3: V2R-PC5
        "#74C476",  # 4: V2R-WiFi
    ]
    cases = ["case1", "case2", "case3", "case4"]

    def build_pct(data):
        df = pd.DataFrame()
        df["case1"] = data["segments_v2n"]
        df["case2"] = data["segments_v2v"]
        df["case3"] = data["segments_v2i_pc5"]
        df["case4"] = data["segments_v2i_wifi"]
        df["total"] = df[cases].sum(axis=1)
        pct = df[cases].div(df["total"], axis=0) * 100
        pct["w1"] = data["w1"]
        return pct

    def plot_traffic_load_area_chart(data, name="MLDRL"):
        pct = build_pct(data)
        x = pct["w1"].values
        knee_w1 = KNEE_W1[name]
        knee_idx = int(np.argmin(np.abs(x - knee_w1)))

        fig, (ax, ax2) = plt.subplots(
            1,
            2,
            figsize=(5, 2.5),
            gridspec_kw={"width_ratios": [2.2, 1.4]},
            sharey=True,
        )

        # ---- main stacked area chart (4 bands, no hatch) ----
        bottom = np.zeros(len(x))
        for i, col in enumerate(cases):
            upper = bottom + pct[col].values
            ax.fill_between(
                x,
                bottom,
                upper,
                alpha=0.4,
                facecolor=colors[i],
                edgecolor=colors[i],
                linewidth=0.1,
            )
            ax.plot(x, upper, color=colors[i], linestyle="-", linewidth=0.5, zorder=3)
            bottom = upper

        ax.axvline(x[knee_idx], color="black", linestyle="--", linewidth=0.5, alpha=0.6)
        ax.set_xlabel(r"Cost weight $w_1$ ($w_2 = 1 - w_1$)")
        ax.set_ylabel("Traffic Percentage (\\%)")
        ax.set_ylim(0, 105)
        ax.set_xticks(x)
        ax.set_xticklabels(x, rotation=90)

        # ---- side panel: share breakdown at w1=0, knee, w1=1 ----
        cols_idx = [0, knee_idx, len(x) - 1]
        col_titles = [
            "$w_1{=}0$",
            f"$w_1{{=}}{x[knee_idx]:.2f}$",
            "$w_1{=}1$",
        ]
        xpos = np.array([0, 1, 2], dtype=float)
        bar_width = 0.5

        for j, idx in enumerate(cols_idx):
            bottom_j = 0.0
            for i, col in enumerate(cases):
                val = pct[col].values[idx]
                ax2.bar(
                    xpos[j],
                    val,
                    bottom=bottom_j,
                    width=bar_width,
                    facecolor=colors[i],
                    edgecolor=colors[i],
                    alpha=0.4,
                    linewidth=0.3,
                )
                if val > 0.5:
                    curly_brace(
                        ax2,
                        bottom_j,
                        bottom_j + val,
                        x=xpos[j] + bar_width / 2 + 0.03,
                        width=0.12,
                        lw=0.5,
                        color=colors[i],
                    )
                    ax2.text(
                        xpos[j] + bar_width / 2 + 0.18,
                        bottom_j + val / 2,
                        f"{val:.1f}\\%",
                        fontsize=4,
                        va="center",
                        ha="left",
                        color=colors[i],
                    )
                bottom_j += val

        ax2.set_xlim(-0.4, 2.75)
        ax2.set_xticks(xpos)
        ax2.set_xticklabels(col_titles, fontsize=6)
        ax2.tick_params(axis="y", labelleft=False)
        ax2.set_ylim(0, 105)

        plt.tight_layout()
        plt.savefig(
            os.path.join(SAVE_DIR, f"{name}_segment_percentage.pdf"),
            bbox_inches="tight",
        )
        plt.close(fig)

    plot_traffic_load_area_chart(WEIGHT_MLDRL, name="MLDRL")
    plot_traffic_load_area_chart(WEIGHT_SLDRL, name="SLDRL")
