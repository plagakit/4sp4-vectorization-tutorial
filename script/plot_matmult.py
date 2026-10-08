import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
JSON_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "logs", "matmult.json")
OUT_PATH = os.path.join(ROOT, "plots", "matmult.png")

medians, stds = {}, {}
with open(JSON_PATH) as f:
	for b in json.load(f)["benchmarks"]:
		if b.get("run_type") != "aggregate":
			continue
		parts = b["run_name"].split("/")
		name = parts[0].removeprefix("BM_").rstrip("_")
		size = " x ".join(parts[1:3])
		if b["aggregate_name"] == "median":
			medians[name] = b
		elif b["aggregate_name"] == "stddev":
			stds[name] = b

metrics = [
	("FLOPs", 1e9, "Median FLOPs", "GFLOP/s"),
	("cpu_time", 1e6, "Median CPU time", "CPU time (ms)"),
]

names = list(medians)
x = np.arange(len(names))
fig, axes = plt.subplots(2, 1, figsize=(10, 11), sharex=True)

for ax, (key, scale, title, ylabel) in zip(axes, metrics):
	med = np.array([medians[n][key] for n in names]) / scale
	std = np.array([stds[n][key] for n in names]) / scale
	ax.bar(x, med, yerr=std, capsize=5, color="tab:blue", edgecolor="black")
	for xi, m, s in zip(x, med, std):
		ax.annotate(f"{m:.2f}", (xi, m + s), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom")
	ax.set_title(title)
	ax.set_ylabel(ylabel)
	ax.set_ylim(0, (med + std).max() * 1.12)
	ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
	ax.grid(axis="y", linestyle="--", alpha=0.5)
	ax.set_axisbelow(True)

axes[-1].set_xticks(x)
axes[-1].set_xticklabels(names, rotation=20, ha="right")
fig.suptitle(f"Matrix-Matrix Multiply ({size})")

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
fig.tight_layout()
fig.savefig(OUT_PATH, dpi=200)