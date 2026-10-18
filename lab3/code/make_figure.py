import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

results = pd.read_csv("../results/S.csv")

sns.boxplot(data=results)
plt.xlabel("Number of clusters (k)")
plt.ylabel("Jaccard similarity")
plt.title("K-means Clustering Stability")

Path("../figs").mkdir(exist_ok=True)
plt.savefig("../figs/stability_plot.png", dpi=300, bbox_inches="tight")
plt.close()