import pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv("results/metrics.csv")
ax = df[df.model=="resnet50"].plot.scatter(x="size_MB", y="accuracy", c="latency_ms", s=120, title="ResNet-50 Pareto")
plt.show()