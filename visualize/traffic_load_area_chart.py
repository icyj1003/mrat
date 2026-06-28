import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import os
from data import WEIGHT_MLDRL, WEIGHT_SLDRL, algo_colors
import pandas as pd

mpl.rcParams["hatch.linewidth"] = 0.5
plt.style.use(["science", "ieee"])


if __name__ == "__main__":
    # Save dir
    SAVE_DIR = "./.output/figures/"
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    labels = [
        "V2B",
        "V2V",
        "V2R-PC5-Cache",
        "V2R-PC5-BS",
        "V2R-WiFi-Cache",
        "V2R-WiFi-BS",
    ]

    colors = [
        "#D6604D",  # Case 1
        "#E69F00",  # Case 2
        "#4393C3",  # Case 3ab
        "#4393C3",  # Case 3c
        "#74C476",  # Case 4ab
        "#74C476",  # Case 4c
    ]

    colors_2 = [
        "#D6604D",  # Case 1
        "#4393C3",  # Case 3c
        "#74C476",  # Case 4c
        "#4393C3",  # Case 3ab
        "#E69F00",  # Case 2
        "#74C476",  # Case 4ab
    ]
    colors_2.reverse()
    colors = colors_2

    hatches = [
        "....",  # Case 1
        "",  # Case 2
        "",  # Case 3ab
        "....",  # Case 3c
        "",  # Case 4ab
        "....",  # Case 4c
    ]

    hatches_2 = [
        "....",  # Case 1
        "....",  # Case 3c
        "....",  # Case 4c
        "",  # Case 2
        "",  # Case 3ab
        "",  # Case 4ab
    ]
    hatches_2.reverse()
    hatches = hatches_2

    def plot_traffic_load_area_chart(data, name="MLDRL"):
        df = pd.DataFrame()
        df["case1"] = data["segments_v2n"]
        df["case2"] = data["segments_v2v"]
        df["case3c"] = data["segments_v2i_pc5_bs_to_edge_to_local"]
        df["case3ab"] = data["segments_v2i_pc5"] - df["case3c"]
        df["case4c"] = data["segments_v2i_wifi_bs_to_edge_to_local"]
        df["case4ab"] = data["segments_v2i_wifi"] - df["case4c"]

        cases = ["case1", "case2", "case3ab", "case3c", "case4ab", "case4c"]
        cases_2 = ["case1", "case3c", "case4c", "case3ab", "case2", "case4ab"]
        cases_2.reverse()
        cases = cases_2

        df["total"] = df[cases].sum(axis=1)
        pct = df[cases].div(df["total"], axis=0) * 100

        x = data["w1"]

        fig, ax = plt.subplots(figsize=(3, 2.5))
        bottom = np.zeros(len(x))

        for i, col in enumerate(cases):
            upper = bottom + pct[col].values

            ax.fill_between(
                x,
                bottom,
                upper,
                alpha=0,
                # facecolor=colors[i],
                # edgecolor="black",
                # -------
                facecolor="white",
                edgecolor=colors[i],
                linewidth=0.1,
                hatch=hatches[i],
            )

            ax.fill_between(
                x,
                bottom,
                upper,
                alpha=0.4,
                # facecolor=colors[i],
                # edgecolor="black",
                # -------
                facecolor=colors[i],
                edgecolor=colors[i],
                linewidth=0.1,
            )

            ax.plot(x, upper, color=colors[i], linestyle="-", linewidth=0.5, zorder=3)

            bottom = upper

            ax.set_xlabel(r"Cost weight $w_1$ ($w_2 = 1 - w_1$)")
            ax.set_ylabel("Traffic Percentage (\\%)")
            ax.set_ylim(0, 105)
            ax.set_xticks(x)
            ax.set_xticklabels(x, rotation=90)
            # ax.legend(frameon=True, loc="center left", fontsize=5)
            plt.tight_layout()
            plt.savefig(
                os.path.join(SAVE_DIR, f"{name}_segment_percentage.pdf"),
                bbox_inches="tight",
            )

    plot_traffic_load_area_chart(WEIGHT_MLDRL, name="MLDRL")
    plot_traffic_load_area_chart(WEIGHT_SLDRL, name="SLDRL")
