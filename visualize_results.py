#!/usr/bin/env python
import os, pandas as pd, matplotlib.pyplot as plt

csv = "results/metrics.csv"
if not os.path.exists(csv):
    raise SystemExit("Run experiments first:  bash scripts/run_all.sh")

df = pd.read_csv(csv)

def plot(sub, title, ylabel):
    plt.figure()
    sc = plt.scatter(sub["size_MB"], sub["accuracy"], c=sub["latency_ms"],
                     s=120, cmap="viridis")
    plt.colorbar(sc, label="Latency (ms)")
    for _, r in sub.iterrows():
        plt.text(r["size_MB"], r["accuracy"], r["technique"], fontsize=8)
    plt.xlabel("Model size (MB)")
    plt.ylabel(ylabel)
    plt.title(title); plt.grid(True); plt.show()

plot(df[df.model=="resnet50"], "ResNet-50 Compression", "Top-1 accuracy (%)")
plot(df[df.model=="bert-base"], "BERT-base Compression", "SST-2 accuracy (%)")