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

# Plot styling (Set canvas dimensions and resolution)
fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
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
ax.set_ylabel("Accuracy (%)", fontsize=10, fontweight="bold")
ax.set_title(
    "Food-101 Accuracy: Supervised Baseline vs. Zero-Shot CLIP",
    fontsize=8,
    fontweight="bold",
    pad=5,  
)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=5)
ax.set_ylim(60, 102)  # Extend upper Y-axis limit to prevent label collision
ax.legend(fontsize=5, loc="lower right")
ax.grid(axis="y", linestyle="--", alpha=0.5)


# Value labels on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            f"{height}%",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )


autolabel(rects1)
autolabel(rects2)

# Automatically adjust layout and save chart with tight bounding box
plt.tight_layout()
plt.savefig(
    "results_chart.png", bbox_inches="tight"
)  # Ensures no cropped margins
plt.show()
