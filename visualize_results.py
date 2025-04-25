#!/usr/bin/env python
import os, pandas as pd, matplotlib.pyplot as plt

csv = "results/metrics.csv"
if not os.path.exists(csv):
    raise SystemExit("Run the experiments first (bash scripts/run_all.sh).")

df = pd.read_csv(csv)

def scatter(sub, title, ylab):
    plt.figure()
    plt.scatter(sub["size_MB"], sub["accuracy"])
    for _, r in sub.iterrows():
        plt.text(r["size_MB"], r["accuracy"], r["technique"], fontsize=8)
    plt.xlabel("Model size (MB)")
    plt.ylabel(ylab)
    plt.title(title)
    plt.grid(True)
    plt.show()

scatter(df[df.model == "resnet50"],
        "ResNet-50 Compression Trade-offs",
        "Top-1 accuracy (%)")

scatter(df[df.model == "bert-base"],
        "BERT-base Compression Trade-offs",
        "SST-2 accuracy (%)")