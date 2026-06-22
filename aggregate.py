import os
import torch
import pandas as pd

dfs = []

cols = [
    "name",
    "cost_per_bit",
    "delay_per_segment",
    "v2i_hit_rate",
    "segments_v2i_wifi",
    "segments_v2i_wifi_remote_to_local",
    "segments_v2i_wifi_bs_to_edge_to_local",
    "segments_v2i_pc5",
    "segments_v2i_pc5_remote_to_local",
    "segments_v2i_pc5_bs_to_edge_to_local",
    "segments_v2v",
    "segments_v2n",
    "avg_activated_links",
    "violation_ratio",
]

for run in os.listdir("./.output/runs"):
    model_path = os.path.join("./.output/runs", run, "model.pth")

    if not os.path.exists(model_path):
        continue

    model = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False,
    )

    metrics = {
        k: v["value"]
        for k, v in model["evaluate"].items()
        if isinstance(v, dict) and "value" in v
    }

    df = pd.DataFrame([metrics])
    df["run"] = run

    old_cols = df.columns.tolist()

    new_cols = cols + [c for c in old_cols if c not in cols]

    # Ignore columns that do not exist
    df = df[[c for c in new_cols if c in df.columns]]

    dfs.append(df)

dataset = pd.concat(dfs, ignore_index=True)

# Build final column order from all columns that actually exist
all_cols = dataset.columns.tolist()
new_cols = cols + [c for c in all_cols if c not in cols]

# Ignore missing columns
dataset = dataset[[c for c in new_cols if c in dataset.columns]]

dataset.to_csv("./.output/out.csv", index=False)

print(f"Saved {len(dataset)} rows to out.csv")
