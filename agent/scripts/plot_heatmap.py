import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Table data input
DATA = [
    [0.15, 0.075, 0.075, 0.1, 0.1, 0.25, 0.2, 0.175],
    [0.1, 0.1, 0.05, 0.1, 0.05, 0.25, 0.1, 0.25],
    [0.1, 0.1, 0.05, 0.05, 0.05, 0.25, 0.1, 0.3],
    [0.15, 0.1, 0.05, 0.1, 0.05, 0.25, 0.15, 0.15],
    [0.15, 0.1, 0.05, 0.1, 0.05, 0.25, 0.1, 0.2],
    [0.1, 0.1, 0.05, 0.1, 0.05, 0.2, 0.1, 0.3],
    [0.15, 0.1, 0.05, 0.1, 0.05, 0.2, 0.1, 0.25],
    [0.15, 0.1, 0.1, 0.15, 0.2, 0.1, 0.1, 0.1],
    [0.15, 0.1, 0.05, 0.2, 0.2, 0.1, 0.05, 0.15]
]
INDEX = [
    "MMLU", "MMLU-Pro", "GSM8K", "MBPP", "HumanEval", "BBH", "HellaSwag", "CNN", "SAMSum"
]
COLUMNS = [
    "Meaning.",
    "Completeness",
    "Expression.",
    "Faithfulness,",
    "Conciseness",
    "Correctness",
    "Structural.",
    "Reasoning."
]

df = pd.DataFrame(DATA, index=INDEX, columns=COLUMNS)

plt.figure(figsize=(8, 6))
ax = sns.heatmap(
    df.T,
    cmap="Reds",
    linewidths=3,
    xticklabels=False,
    yticklabels=False,
    annot=False,
    square=True
)
# Increase colorbar (side numbers) font size by 2x
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=16)  # If default is 12, then 24
plt.xticks(fontsize=12, rotation=45, ha='right')
plt.yticks(fontsize=12, rotation=0)
plt.xlabel("")
plt.ylabel("")
plt.tight_layout()
plt.savefig("heatmap.png", dpi=300, bbox_inches='tight')
plt.show() 