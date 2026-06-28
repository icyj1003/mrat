import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import os
from data import WEIGHT_MLDRL, WEIGHT_SLDRL, algo_colors

plt.style.use(["science", "ieee"])

if __name__ == "__main__":
    # Save dir
    SAVE_DIR = "./.output/figures/"
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # Representative point
    KNEE_ML = 0.55
    KNEE_SL = 0.30

    # Load data
    cost = np.array(WEIGHT_MLDRL["cost_per_bit"])
    delay = np.array(WEIGHT_MLDRL["delay_per_segment"])
    cost1 = np.array(WEIGHT_SLDRL["cost_per_bit"])
    delay1 = np.array(WEIGHT_SLDRL["delay_per_segment"])
    w = np.array(WEIGHT_MLDRL["w1"], dtype=float)

    def pareto_front(cost, delay):
        points = np.stack([cost, delay], axis=1)
        is_pareto = np.ones(len(points), dtype=bool)
        for i, p in enumerate(points):
            if not is_pareto[i]:
                continue
            dominated = np.all(points <= p, axis=1) & np.any(points < p, axis=1)
            if np.any(dominated):
                is_pareto[i] = False
        return is_pareto

    mask_ml = pareto_front(cost, delay)
    mask_sl = pareto_front(cost1, delay1)

    # Extract Pareto points
    pf_cost_ml = cost[mask_ml]
    pf_delay_ml = delay[mask_ml]
    pf_w_ml = w[mask_ml]

    idx_ml = np.argsort(pf_cost_ml)
    pf_cost_ml = pf_cost_ml[idx_ml]
    pf_delay_ml = pf_delay_ml[idx_ml]
    pf_w_ml = pf_w_ml[idx_ml]

    pf_cost_sl = cost1[mask_sl]
    pf_delay_sl = delay1[mask_sl]
    pf_w_sl = w[mask_sl]

    idx_sl = np.argsort(pf_cost_sl)
    pf_cost_sl = pf_cost_sl[idx_sl]
    pf_delay_sl = pf_delay_sl[idx_sl]
    pf_w_sl = pf_w_sl[idx_sl]

    # Locate the knee point
    knee_idx_ml = np.argmin(np.abs(pf_w_ml - KNEE_ML))
    knee_idx_sl = np.argmin(np.abs(pf_w_sl - KNEE_SL))

    # Plotting
    plt.figure()

    # --- all solutions ---
    plt.scatter(
        cost,
        delay,
        color=algo_colors["MLDRL"],
        alpha=0.15,
        s=20,
        label="MLDRL Solutions",
    )
    plt.scatter(
        cost1,
        delay1,
        color=algo_colors["SLDRL"],
        alpha=0.15,
        s=20,
        label="SLDRL Solutions",
    )

    # --- Pareto fronts ---
    plt.plot(
        pf_cost_ml,
        pf_delay_ml,
        "-v",
        color=algo_colors["MLDRL"],
        linewidth=1,
        markersize=5,
        label="MLDRL Pareto Front",
    )
    plt.plot(
        pf_cost_sl,
        pf_delay_sl,
        "-D",
        color=algo_colors["SLDRL"],
        linewidth=1,
        markersize=5,
        label="SLDRL Pareto Front",
    )

    # --- Weight annotations (skip the knee — already marked) ---
    for i, (x, y, weight) in enumerate(zip(pf_cost_ml, pf_delay_ml, pf_w_ml)):
        plt.annotate(
            f"{weight:.2f}",
            (x, y),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )

    for i, (x, y, weight) in enumerate(zip(pf_cost_sl, pf_delay_sl, pf_w_sl)):
        plt.annotate(
            f"{weight:.2f}",
            (x, y),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )

    plt.xlabel("Average Cost per bit")
    plt.ylabel("Average Delay per segment (ms)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=5)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "pareto.pdf"))
