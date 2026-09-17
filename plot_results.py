import matplotlib.pyplot as plt
import numpy as np

# Data setup
models = [
    "ResNet-50\n(Supervised)",
    "CLIP (ViT-B/32)\n(Basic Prompt)",
    "CLIP (ViT-B/32)\n(Prompt Ensemble)",
]
top1_acc = [84.1, 78.2, 82.5]
top5_acc = [96.2, 93.5, 95.8]

x = np.arange(len(models))
width = 0.35

# Plot styling
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
rects1 = ax.bar(
    x - width / 2,
    top1_acc,
    width,
    label="Top-1 Accuracy (%)",
    color="#1f77b4",
)
rects2 = ax.bar(
    x + width / 2,
    top5_acc,
    width,
    label="Top-5 Accuracy (%)",
    color="#aec7e8",
)

# Formatting
ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
ax.set_title(
    "Food-101 Accuracy: Supervised Baseline vs. Zero-Shot CLIP",
    fontsize=14,
    fontweight="bold",
    pad=15,
)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.set_ylim(60, 100)
ax.legend(fontsize=11, loc="lower right")
ax.grid(axis="y", linestyle="--", alpha=0.5)


# Value labels on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            f"{height}%",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )


autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig("results_chart.png")
plt.show()
