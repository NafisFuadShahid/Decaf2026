"""Step 7: Write VERDICT.md and print to stdout."""
import json
import sys
from pathlib import Path
from datetime import datetime


def main():
    print("=" * 60)
    print("STEP 7: WRITE VERDICT")
    print("=" * 60)

    with open("C:/DeCaf/fed_crc_results/step5_crc_results.json") as f:
        full_results = json.load(f)
    with open("C:/DeCaf/fed_crc_results/step1_status.json") as f:
        step1 = json.load(f)
    with open("C:/DeCaf/fed_crc_results/step2_subjects.json") as f:
        step2 = json.load(f)
    with open("C:/DeCaf/fed_crc_results/step3_model.json") as f:
        step3 = json.load(f)
    with open("C:/DeCaf/fed_crc_results/step4_status.json") as f:
        step4 = json.load(f)

    summary = full_results["summary"]
    alpha = full_results["alpha"]
    eval_results = full_results["eval_results"]
    alpha_sweep = full_results["alpha_sweep"]

    verdict = summary["verdict"]
    rationale = summary["rationale"]

    K = summary["n_sites"]
    n_subjects = step4["n_subjects_with_tumor"]
    data_source = step1["data_source"]
    partition_source = step2["partition_source"]
    model_name = step3["model_name"]

    # Deduplicate fallbacks and errors
    seen_fb = set()
    all_fallbacks = []
    for fb in (
        list(step1.get("fallbacks", []))
        + list(step2.get("fallbacks", []))
        + list(step3.get("fallbacks", []))
        + list(step4.get("fallbacks", []))
        + list(full_results.get("fallbacks", []))
    ):
        if fb not in seen_fb:
            seen_fb.add(fb)
            all_fallbacks.append(fb)

    seen_err = set()
    all_errors = []
    for err in (
        list(step1.get("errors", []))
        + list(step2.get("errors", []))
        + list(step3.get("errors", []))
        + list(step4.get("errors", []))
        + list(full_results.get("errors", []))
    ):
        if err not in seen_err:
            seen_err.add(err)
            all_errors.append(err)

    # Per-method stats
    method_stats = {}
    for method in eval_results:
        all_fnrs = []
        worst_fnr = 0
        mean_stretch = []
        violations = 0
        for sid_str, res in eval_results[method].items():
            all_fnrs.extend(res["per_case_fnrs"])
            mean_stretch.append(res["mean_stretch"])
            if res["mean_fnr"] > worst_fnr:
                worst_fnr = res["mean_fnr"]
            if res["mean_fnr"] > alpha:
                violations += 1

        import numpy as np
        method_stats[method] = {
            "marginal": float(np.mean(all_fnrs)) if all_fnrs else -1,
            "worst": float(worst_fnr),
            "violations": int(violations),
            "stretch": float(np.mean(mean_stretch)) if mean_stretch else -1,
        }

    # Figure assessment
    b3_worst = summary["b3_worst_fnr"]
    ours_worst = summary["ours_worst_fnr"]
    b3_marg = summary["b3_marginal_fnr"]
    ours_marg = summary["ours_marginal_fnr"]
    separation = summary["separation"]

    b3_violations = summary["b3_violations"]
    ours_violations_per_site = summary["ours_violations"]

    if separation >= 0.05 or (separation >= 0.03 and b3_violations >= K // 4):
        fig_assessment = (
            f"CLEAR VISUAL SEPARATION (GREEN). "
            f"B3 worst-site FNR = {b3_worst:.3f} (exceeds α by {separation:.1%}). "
            f"B3 fails {b3_violations}/{K} sites while our per-site method fails {ours_violations_per_site}/{K}. "
            f"The figure shows B3 crossing above the α={alpha} dashed line for multiple sites "
            f"while our method stays below (or barely at) the coverage boundary."
        )
    elif separation >= 0.02:
        fig_assessment = (
            f"MODERATE SEPARATION. B3 worst-site FNR = {b3_worst:.3f} "
            f"(exceeds α by {separation:.1%}). "
            f"Some visual separation present but story may need strengthening."
        )
    else:
        fig_assessment = (
            f"INSUFFICIENT SEPARATION. B3 worst-site FNR = {b3_worst:.3f} "
            f"(barely exceeds α by {separation:.1%}). "
            f"The failure mode is not visually compelling."
        )

    # Alpha sweep summary
    sweep_lines = []
    for a_str in sorted(alpha_sweep.keys(), key=float):
        a = float(a_str)
        b3_w = alpha_sweep[a_str]["B3"]["worst_fnr"]
        ours_w = alpha_sweep[a_str]["Ours_shared"]["worst_fnr"]
        b3_v = alpha_sweep[a_str]["B3"]["violations"]
        ours_v = alpha_sweep[a_str]["Ours_shared"]["violations"]
        sweep_lines.append(
            f"  α={a:.2f}: B3 worst={b3_w:.3f} ({b3_v} viol.), "
            f"Ours worst={ours_w:.3f} ({ours_v} viol.)"
        )

    verdict_text = f"""# Fed-CRC-Seg Pilot Verdict

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Final Decision: {verdict}

## Rationale
{rationale}

---

## Key Numbers (α = {alpha})

| Method | Marginal FNR | Worst-site FNR | Sites Violating α | Avg Stretch |
|--------|-------------|----------------|-------------------|-------------|
| B1 Centralized Oracle | {method_stats.get('B1_centralized', {}).get('marginal', -1):.4f} | {method_stats.get('B1_centralized', {}).get('worst', -1):.4f} | {method_stats.get('B1_centralized', {}).get('violations', -1)}/{K} | {method_stats.get('B1_centralized', {}).get('stretch', -1):.3f} |
| B2 Per-site Local CRC | {method_stats.get('B2_per_site', {}).get('marginal', -1):.4f} | {method_stats.get('B2_per_site', {}).get('worst', -1):.4f} | {method_stats.get('B2_per_site', {}).get('violations', -1)}/{K} | {method_stats.get('B2_per_site', {}).get('stretch', -1):.3f} |
| B3 Naive Pooled CRC | {method_stats.get('B3_naive_pooled', {}).get('marginal', -1):.4f} | {method_stats.get('B3_naive_pooled', {}).get('worst', -1):.4f} | {method_stats.get('B3_naive_pooled', {}).get('violations', -1)}/{K} | {method_stats.get('B3_naive_pooled', {}).get('stretch', -1):.3f} |
| Ours (per-site λ) | {method_stats.get('Ours_per_site', {}).get('marginal', -1):.4f} | {method_stats.get('Ours_per_site', {}).get('worst', -1):.4f} | {method_stats.get('Ours_per_site', {}).get('violations', -1)}/{K} | {method_stats.get('Ours_per_site', {}).get('stretch', -1):.3f} |
| **Ours (shared λ)** | **{method_stats.get('Ours_shared', {}).get('marginal', -1):.4f}** | **{method_stats.get('Ours_shared', {}).get('worst', -1):.4f}** | **{method_stats.get('Ours_shared', {}).get('violations', -1)}/{K}** | **{method_stats.get('Ours_shared', {}).get('stretch', -1):.3f}** |

---

## Failure Mode Separation
- **B3 worst-site FNR above α:** {separation*100:.1f} percentage points
- **B3 sites violating α:** {summary['b3_violations']}/{K}
- **Our method worst-site FNR above α:** {(ours_worst - alpha)*100:.1f} pp
- **Our method sites violating α:** {summary['ours_violations']}/{K}

---

## Figure 2 Assessment
{fig_assessment}

---

## Alpha Sweep
{chr(10).join(sweep_lines)}

---

## Data Used
- **Dataset:** {data_source}
- **Partition:** {partition_source} institutional partitions
- **Model:** {model_name}
- **Subjects with tumor (used for CRC):** {n_subjects}
- **Sites used (≥6 subjects each):** {K}

---

## Errors and Fallbacks
{chr(10).join('- ' + f for f in all_fallbacks) if all_fallbacks else "- None"}

**Errors:**
{chr(10).join('- ' + e for e in all_errors) if all_errors else "- None"}

---

## Interpretation
```
GREEN  → Proceed to write the DeCaF 2026 paper.
         The failure mode is real, large, and our fix works.
YELLOW → Investigate further. Failure mode exists but is not large enough
         to be the paper's primary result. Consider: different model weights,
         more heterogeneous data split, or reframe as an existence proof.
RED    → The failure mode doesn't manifest with this setup.
         Either the model is too uncertain everywhere (high FNR at all λ),
         sites are too homogeneous, or the prediction quality is insufficient.
         Kill or substantially redesign the idea.
```

**Current verdict: {verdict}**
"""

    out_path = "C:/DeCaf/fed_crc_results/VERDICT.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(verdict_text)

    print(verdict_text)
    print(f"\nVERDICT.md saved to {out_path}")


if __name__ == "__main__":
    main()
