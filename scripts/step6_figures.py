"""Step 6: Generate Figure 2 and summary plots."""
import os
import sys
import json
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from collections import defaultdict


def make_figure2(full_results, volume_scores):
    """Generate the main money figure: per-site FNR and set size."""
    alpha = full_results["alpha"]
    eval_results = full_results["eval_results"]
    valid_sites = full_results["valid_sites"]

    # Compute mean tumor volume per site for ordering
    site_tumor_vol = {}
    for site_id_str, subjects_list in valid_sites.items():
        site_id = int(site_id_str)
        vols = [volume_scores[s]["tumor_volume"]
                for s in subjects_list if s in volume_scores]
        site_tumor_vol[site_id] = np.mean(vols) if vols else 0

    # Sort sites by ascending tumor prevalence
    sorted_sites = sorted(site_tumor_vol.keys(), key=lambda s: site_tumor_vol[s])
    x_positions = np.arange(len(sorted_sites))

    fig = plt.figure(figsize=(16, 7))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    style = {
        "B3_naive_pooled": {"color": "#E74C3C", "marker": "o", "lw": 2.2, "label": "Naive Pooled CRC (B3)", "zorder": 4},
        "B2_per_site":     {"color": "#F39C12", "marker": "^", "lw": 1.5, "label": "Per-site Local CRC (B2)", "zorder": 3},
        "Ours_shared":     {"color": "#27AE60", "marker": "D", "lw": 2.2, "label": "Ours (site-conditional λ)", "zorder": 5},
        "B1_centralized":  {"color": "#2980B9", "marker": "s", "lw": 1.5, "label": "Centralized Oracle (B1)", "zorder": 2},
    }

    plot_order = ["B1_centralized", "B2_per_site", "B3_naive_pooled", "Ours_shared"]

    for method in plot_order:
        if method not in eval_results:
            continue
        fnr_vals = []
        stretch_vals = []
        for sid in sorted_sites:
            sid_str = str(sid)
            if sid_str in eval_results[method]:
                fnr_vals.append(eval_results[method][sid_str]["mean_fnr"])
                stretch_vals.append(eval_results[method][sid_str]["mean_stretch"])
            else:
                fnr_vals.append(np.nan)
                stretch_vals.append(np.nan)

        s = style[method]
        ax1.plot(x_positions, fnr_vals,
                 color=s["color"], marker=s["marker"], label=s["label"],
                 linewidth=s["lw"], markersize=7, alpha=0.9, zorder=s["zorder"])
        ax2.plot(x_positions, stretch_vals,
                 color=s["color"], marker=s["marker"], label=s["label"],
                 linewidth=s["lw"], markersize=7, alpha=0.9, zorder=s["zorder"])

    # Alpha line
    ax1.axhline(y=alpha, color="black", linestyle="--", linewidth=1.8,
                label=f"α = {alpha}", zorder=6)

    ax1.set_xlabel("Site index (↑ tumor prevalence)", fontsize=13)
    ax1.set_ylabel("Empirical FNR on held-out test set", fontsize=13)
    ax1.set_title("Per-site Coverage (FNR)", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=9.5, loc="upper left", framealpha=0.9)
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels([str(s) for s in sorted_sites], rotation=45, fontsize=9)
    ax1.grid(axis="y", alpha=0.3, linestyle=":")
    ax1.set_ylim(-0.02, min(1.05, ax1.get_ylim()[1] + 0.05))

    ax2.set_xlabel("Site index (↑ tumor prevalence)", fontsize=13)
    ax2.set_ylabel("Avg prediction set size / |GT tumor|", fontsize=13)
    ax2.set_title("Per-site Prediction Set Size", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=9.5, loc="upper right", framealpha=0.9)
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels([str(s) for s in sorted_sites], rotation=45, fontsize=9)
    ax2.grid(axis="y", alpha=0.3, linestyle=":")

    plt.suptitle("Site-Conditional CRC vs. Naive Pooled Calibration\n"
                 "Multi-institutional Brain Tumor Segmentation (FeTS/BraTS)",
                 fontsize=13, y=1.01)

    out_dir = Path("C:/DeCaf/fed_crc_results")
    plt.savefig(str(out_dir / "figure2_failure_mode.png"), dpi=300, bbox_inches="tight")
    plt.savefig(str(out_dir / "figure2_failure_mode.pdf"), bbox_inches="tight")
    plt.close()
    print("Saved Figure 2 (PNG + PDF)")


def make_alpha_sweep_figure(full_results):
    """Generate alpha sweep plot."""
    alpha_sweep = full_results["alpha_sweep"]

    alpha_vals = sorted(float(a) for a in alpha_sweep.keys())

    b3_worst = [alpha_sweep[str(a)]["B3"]["worst_fnr"] for a in alpha_vals]
    ours_worst = [alpha_sweep[str(a)]["Ours_shared"]["worst_fnr"] for a in alpha_vals]
    b3_marg = [alpha_sweep[str(a)]["B3"]["marginal_fnr"] for a in alpha_vals]
    ours_marg = [alpha_sweep[str(a)]["Ours_shared"]["marginal_fnr"] for a in alpha_vals]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(alpha_vals, b3_worst, "ro-", label="Naive Pooled (B3) — worst site", lw=2)
    ax1.plot(alpha_vals, ours_worst, "gs-", label="Ours (shared λ) — worst site", lw=2)
    ax1.plot(alpha_vals, alpha_vals, "k--", label="Target α (ideal)", lw=1.5)
    ax1.set_xlabel("Target coverage α", fontsize=12)
    ax1.set_ylabel("Worst-site FNR", fontsize=12)
    ax1.set_title("Worst-site FNR vs. Target α", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    ax2.plot(alpha_vals, b3_marg, "ro-", label="Naive Pooled (B3) — marginal", lw=2)
    ax2.plot(alpha_vals, ours_marg, "gs-", label="Ours (shared λ) — marginal", lw=2)
    ax2.plot(alpha_vals, alpha_vals, "k--", label="Target α (ideal)", lw=1.5)
    ax2.set_xlabel("Target coverage α", fontsize=12)
    ax2.set_ylabel("Marginal FNR", fontsize=12)
    ax2.set_title("Marginal FNR vs. Target α", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("C:/DeCaf/fed_crc_results/figure_alpha_sweep.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved alpha sweep figure")


def make_lambda_curves_figure(full_results):
    """Plot per-site calibration risk curves."""
    site_risk_curves = full_results["site_risk_curves"]
    lambda_grid = full_results["lambda_grid"]
    alpha = full_results["alpha"]

    fig, ax = plt.subplots(figsize=(10, 6))

    cmap = plt.get_cmap("tab10")
    for i, (sid_str, risk) in enumerate(site_risk_curves.items()):
        ax.plot(lambda_grid, risk, color=cmap(i % 10),
                label=f"Site {sid_str}", alpha=0.75, lw=1.5)

    ax.axhline(y=alpha, color="black", linestyle="--", lw=2, label=f"α = {alpha}")
    ax.set_xlabel("λ (threshold parameter)", fontsize=12)
    ax.set_ylabel("Empirical risk R_k(λ) = E[FNR] + B/(n+1)", fontsize=12)
    ax.set_title("Per-site CRC Risk Curves", fontsize=13)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("C:/DeCaf/fed_crc_results/figure_risk_curves.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved risk curves figure")


def make_summary_table(full_results):
    """Save summary results as CSV."""
    import pandas as pd

    alpha = full_results["alpha"]
    eval_results = full_results["eval_results"]
    summary = full_results["summary"]

    rows = []
    for method in eval_results:
        all_fnrs = []
        worst_fnr = 0
        worst_site = None
        mean_stretch = []
        violations = 0

        for sid_str, res in eval_results[method].items():
            all_fnrs.extend(res["per_case_fnrs"])
            mean_stretch.append(res["mean_stretch"])
            if res["mean_fnr"] > worst_fnr:
                worst_fnr = res["mean_fnr"]
                worst_site = sid_str
            if res["mean_fnr"] > alpha:
                violations += 1

        n_sites = len(eval_results[method])
        rows.append({
            "Method": method,
            "Marginal FNR": round(float(np.mean(all_fnrs)), 4) if all_fnrs else -1,
            "Worst-site FNR": round(float(worst_fnr), 4),
            "Worst Site": worst_site,
            "Sites Violating α": violations,
            "N Sites": n_sites,
            "Mean Set Size (stretch)": round(float(np.mean(mean_stretch)), 3) if mean_stretch else -1,
        })

    df = pd.DataFrame(rows)
    print("\nSummary Table:")
    print(df.to_string(index=False))

    df.to_csv("C:/DeCaf/fed_crc_results/summary_table.csv", index=False)
    print("Saved summary_table.csv")

    return df


def main():
    print("=" * 60)
    print("STEP 6: GENERATE FIGURES")
    print("=" * 60)

    with open("C:/DeCaf/fed_crc_results/step5_crc_results.json") as f:
        full_results = json.load(f)
    with open("C:/DeCaf/fed_crc_results/volume_scores.pkl", "rb") as f:
        volume_scores = pickle.load(f)

    make_figure2(full_results, volume_scores)
    make_alpha_sweep_figure(full_results)
    make_lambda_curves_figure(full_results)
    df = make_summary_table(full_results)

    print("\nSTEP 6 COMPLETE — all figures and tables saved to C:/DeCaf/fed_crc_results/")


if __name__ == "__main__":
    main()
