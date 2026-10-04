import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def clean_name(name):
	# "BM_matmul_base_/1000/1000/iterations:1/repeats:20" -> "matmul_base"
	name = name.split("/")[0]
	if name.startswith("BM_"):
		name = name[3:]
	return name.rstrip("_")


def load_flops(path):
	with open(path) as f:
		data = json.load(f)
	results = {}
	for b in data["benchmarks"]:
		if b.get("run_type", "iteration") != "iteration":
			continue
		if "FLOPs" not in b:
			continue
		name = clean_name(b.get("run_name", b["name"]))
		results.setdefault(name, []).append(b["FLOPs"])
	return results


def main():
	json_path = sys.argv[1]
	if len(sys.argv) > 2:
		out_path = sys.argv[2]
	else:
		base = os.path.splitext(os.path.basename(json_path))[0]
		os.makedirs("plots", exist_ok=True)
		out_path = os.path.join("plots", f"{base}-flops.png")

	results = load_flops(json_path)
	if not results:
		print("No benchmarks with a 'FLOPs' counter found in", json_path)
		sys.exit(1)

	names = list(results.keys())
	means = np.array([np.mean(results[n]) for n in names])
	stds = np.array([np.std(results[n]) for n in names])

	fig, ax = plt.subplots(figsize=(10, 6))
	x = np.arange(len(names))
	bars = ax.bar(x, means, yerr=stds, capsize=5, color="tab:blue", edgecolor="black")

	for bar, m in zip(bars, means):
		ax.annotate(
			f"{m / 1e9:.2f}",
			(bar.get_x() + bar.get_width() / 2, m),
			xytext=(0, 4),
			textcoords="offset points",
			ha="center",
			va="bottom",
			fontsize=9,
		)

	ax.set_xticks(x)
	ax.set_xticklabels(names, rotation=20, ha="right")
	ax.set_ylabel("Performance (GFLOP/s)")
	ax.set_title("Matrix-vector multiply: FLOP/s by implementation (1000 x 1000)")
	ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v / 1e9:g}"))
	ax.set_ylim(bottom=0)
	ax.grid(axis="y", linestyle="--", alpha=0.5)
	ax.set_axisbelow(True)

	fig.tight_layout()
	fig.savefig(out_path, dpi=200)
	print("Saved", out_path)


if __name__ == "__main__":
	main()