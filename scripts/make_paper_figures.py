"""
Generate additional paper-ready figures from REAL, already-computed experiment data
(training logs, evaluation JSONs) -- no new experiments, no fabricated numbers. Every
number plotted here is read at runtime from the same files already cited throughout
docs/design_decisions.md, docs/internvl3_experiment.md, and results/metrics_table.md.

Usage:
    python scripts/make_paper_figures.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "paper_figures")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

COLOR_BASELINE = "#9e9e9e"
COLOR_TEXT_LORA = "#5b8fd1"
COLOR_VISION_LORA = "#d1495b"
COLOR_INTERNVL3 = "#2a9d8f"
COLOR_REF = "#333333"


def load_json(path):
    with open(os.path.join(ROOT, path), encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path):
    with open(os.path.join(ROOT, path), encoding="utf-8") as f:
        return [json.loads(l) for l in f]


# --------------------------------------------------------------------------------------
# Fig 1: BLIP training convergence, 3 phases, 30 epochs (real per-epoch val_loss)
# --------------------------------------------------------------------------------------
def fig_training_convergence():
    v1 = load_jsonl("logs/full_run_v1_epochs.jsonl")
    v2 = load_jsonl("logs/full_run_v2_epochs.jsonl")
    v3 = load_jsonl("logs/full_run_v3_vision_epochs.jsonl")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    phases = [("Fase 1: LoRA teks (epoch 1-10)", v1, 0, COLOR_TEXT_LORA),
              ("Fase 2: lanjutan LoRA teks (epoch 11-20)", v2, 10, "#3a6fa8"),
              ("Fase 3: + LoRA vision (epoch 21-30)", v3, 20, COLOR_VISION_LORA)]
    for label, data, offset, color in phases:
        epochs = [offset + r["epoch"] for r in data]
        losses = [r["val_loss"] for r in data]
        ax.plot(epochs, losses, marker="o", markersize=4, label=label, color=color, linewidth=2)

    ax.axvline(10.5, color="gray", linestyle=":", linewidth=1)
    ax.axvline(20.5, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Epoch kumulatif")
    ax.set_ylabel("Validation loss")
    ax.set_title("Konvergensi training BLIP (30 epoch, 3 fase)")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)
    final_loss = v3[-1]["val_loss"]
    ax.annotate(f"val_loss final = {final_loss:.4f}", xy=(30, final_loss),
                xytext=(22, final_loss + 0.15), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="black", lw=1))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig01_training_convergence.png"))
    fig.savefig(os.path.join(OUT_DIR, "fig01_training_convergence.pdf"))
    plt.close(fig)
    print("[fig01] saved -- final val_loss (30ep):", final_loss)


# --------------------------------------------------------------------------------------
# Fig 2: main results, 3-way comparison, real metrics.json
# --------------------------------------------------------------------------------------
def fig_main_results():
    m = load_json("results/metrics.json")
    base = m["baseline_zero_shot_blip_translated"]
    txt = m["ablation_text_only_lora"]
    vis = m["final_text_plus_vision_lora"]

    metrics = ["cider", "bleu4", "rougeL", "bertscore_f1_indolem"]
    labels = ["CIDEr", "BLEU-4", "ROUGE-L", "BERTScore F1"]
    base_vals = [base[k] for k in metrics]
    txt_vals = [txt[k] for k in metrics]
    vis_vals = [vis[k] for k in metrics]

    x = np.arange(len(metrics))
    w = 0.26
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(x - w, base_vals, w, label="Baseline zero-shot", color=COLOR_BASELINE)
    ax.bar(x, txt_vals, w, label="LoRA teks-saja", color=COLOR_TEXT_LORA)
    ax.bar(x + w, vis_vals, w, label="LoRA teks+vision (final)", color=COLOR_VISION_LORA)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Skor")
    ax.set_title(f"Hasil evaluasi utama (n={m['n_paired_examples_scored']} data uji)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    for xi, v in zip(x + w, vis_vals):
        ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig02_main_results_comparison.png"))
    fig.savefig(os.path.join(OUT_DIR, "fig02_main_results_comparison.pdf"))
    plt.close(fig)
    print("[fig02] saved -- CIDEr baseline/text/vision:", base_vals[0], txt_vals[0], vis_vals[0])


# --------------------------------------------------------------------------------------
# Fig 3: bias diagnostic before/after vision-LoRA
# --------------------------------------------------------------------------------------
def fig_bias_diagnostic():
    v = load_json("results/vision_vs_baseline.json")
    before = v["baseline_text_only_lora"]
    after = v["vision_lora"]
    ref_no_damage = after["building_no_damage_rate_reference"]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    ax = axes[0]
    vals = [before["bencana_exact_match_rate"] * 100, after["bencana_exact_match_rate"] * 100]
    bars = ax.bar(["LoRA teks-saja", "+ LoRA vision"], vals,
                   color=[COLOR_TEXT_LORA, COLOR_VISION_LORA])
    ax.set_ylabel("Akurasi (%)")
    ax.set_title("Akurasi klasifikasi jenis bencana")
    ax.set_ylim(0, 100)
    for b, val in zip(bars, vals):
        ax.annotate(f"{val:.1f}%", (b.get_x() + b.get_width() / 2, val),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    vals = [before["building_no_damage_rate_generated"] * 100,
            after["building_no_damage_rate_generated"] * 100]
    bars = ax.bar(["LoRA teks-saja", "+ LoRA vision"], vals,
                   color=[COLOR_TEXT_LORA, COLOR_VISION_LORA])
    ax.axhline(ref_no_damage * 100, color=COLOR_REF, linestyle="--", linewidth=1.5,
               label=f"Rate referensi asli ({ref_no_damage*100:.1f}%)")
    ax.set_ylabel("Rate (%)")
    ax.set_title('Rate prediksi "tidak ada kerusakan"')
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8)
    for b, val in zip(bars, vals):
        ax.annotate(f"{val:.1f}%", (b.get_x() + b.get_width() / 2, val),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Efek penambahan LoRA vision terhadap bias model", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig03_bias_diagnostic.png"))
    fig.savefig(os.path.join(OUT_DIR, "fig03_bias_diagnostic.pdf"))
    plt.close(fig)
    print("[fig03] saved -- BENCANA acc before/after:", before["bencana_exact_match_rate"], after["bencana_exact_match_rate"])


# --------------------------------------------------------------------------------------
# Fig 4: BLIP vs InternVL3 training efficiency (real wall-clock from logs)
# --------------------------------------------------------------------------------------
def fig_efficiency_comparison():
    v1 = load_jsonl("logs/full_run_v1_epochs.jsonl")
    v2 = load_jsonl("logs/full_run_v2_epochs.jsonl")
    v3 = load_jsonl("logs/full_run_v3_vision_epochs.jsonl")
    blip_total_s = v1[-1]["elapsed_total_s"] + v2[-1]["elapsed_total_s"] + v3[-1]["elapsed_total_s"]
    blip_epochs = 30
    blip_per_epoch_min = blip_total_s / blip_epochs / 60

    iv1 = load_jsonl("logs/internvl3_run1_epochs.jsonl")
    iv2 = load_jsonl("logs/internvl3_run1_ep2_epochs.jsonl")
    iv_total_s = iv1[-1]["elapsed_total_s"] + iv2[-1]["elapsed_total_s"]
    iv_epochs = 2
    iv_per_epoch_min = iv_total_s / iv_epochs / 60

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4))

    ax = axes[0]
    bars = ax.bar(["BLIP\n(224M)", "InternVL3\n(938M)"], [blip_per_epoch_min, iv_per_epoch_min],
                   color=[COLOR_VISION_LORA, COLOR_INTERNVL3])
    ax.set_ylabel("Menit / epoch")
    ax.set_title("Waktu wall-clock per epoch")
    for b, val in zip(bars, [blip_per_epoch_min, iv_per_epoch_min]):
        ax.annotate(f"{val:.1f} min", (b.get_x() + b.get_width() / 2, val),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=9)
    ratio = iv_per_epoch_min / blip_per_epoch_min
    ax.text(0.5, 0.92, f"InternVL3 ~{ratio:.1f}x lebih lambat/epoch", transform=ax.transAxes,
            ha="center", fontsize=9, style="italic")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    bars = ax.bar(["BLIP", "InternVL3"], [blip_total_s / 3600, iv_total_s / 3600],
                   color=[COLOR_VISION_LORA, COLOR_INTERNVL3])
    ax.set_ylabel("Total jam wall-clock")
    ax.set_title(f"Total waktu: {blip_epochs} epoch (BLIP, konvergen)\nvs {iv_epochs} epoch (InternVL3, belum konvergen)")
    for b, val, n in zip(bars, [blip_total_s / 3600, iv_total_s / 3600], [blip_epochs, iv_epochs]):
        ax.annotate(f"{val:.2f}j\n({n} epoch)", (b.get_x() + b.get_width() / 2, val),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Efisiensi training: BLIP vs InternVL3-1B (GPU identik, RTX 5050 8GB)", y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig04_internvl3_efficiency.png"))
    fig.savefig(os.path.join(OUT_DIR, "fig04_internvl3_efficiency.pdf"))
    plt.close(fig)
    print(f"[fig04] saved -- BLIP {blip_per_epoch_min:.2f} min/ep, InternVL3 {iv_per_epoch_min:.2f} min/ep, ratio {ratio:.2f}x")


# --------------------------------------------------------------------------------------
# Fig 5: InternVL3 epoch1/epoch2 vs BLIP final -- CIDEr/BLEU-4 + significance
# --------------------------------------------------------------------------------------
def fig_internvl3_vs_blip():
    d = load_json("results/internvl3_epoch1_vs_epoch2_vs_blip.json")
    ep1, ep2, blip = d["internvl3_epoch1"], d["internvl3_epoch2"], d["blip_vision_lora_final"]

    metrics = ["cider", "bleu4", "rougeL", "bertscore_f1"]
    labels = ["CIDEr", "BLEU-4", "ROUGE-L", "BERTScore F1"]
    x = np.arange(len(metrics))
    w = 0.26
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    ax.bar(x - w, [ep1[k] for k in metrics], w, label="InternVL3 epoch 1", color="#a8dadc")
    ax.bar(x, [ep2[k] for k in metrics], w, label="InternVL3 epoch 2", color=COLOR_INTERNVL3)
    ax.bar(x + w, [blip[k] for k in metrics], w, label="BLIP final (30 epoch)", color=COLOR_VISION_LORA)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Skor")
    ax.set_title(f"InternVL3 (n={d['n_paired']}) vs BLIP final -- 2.362 data uji")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    sig = d["significance_epoch2_vs_blip_cider"]
    ax.text(0.02, 0.95, f"epoch2 vs BLIP (CIDEr): diff={sig['observed_mean_diff']:+.4f}, p={sig['two_sided_p_value_approx']:.3f} (n.s.)",
            transform=ax.transAxes, fontsize=8, va="top", style="italic")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig05_internvl3_vs_blip_results.png"))
    fig.savefig(os.path.join(OUT_DIR, "fig05_internvl3_vs_blip_results.pdf"))
    plt.close(fig)
    print("[fig05] saved -- epoch2 vs blip CIDEr diff:", sig["observed_mean_diff"], "p:", sig["two_sided_p_value_approx"])


# --------------------------------------------------------------------------------------
# Fig 6: xAI faithfulness -- 4-method comparison (rollout/last_layer/random/rise)
# --------------------------------------------------------------------------------------
def fig_xai_faithfulness():
    d = load_json("results/xai_rise/deletion_insertion_test.json")
    auc = d["auc"]
    sig = d["significance_paired_bootstrap"]["token_f1"]

    methods = ["random", "rollout", "last_layer", "rise"]
    method_labels = ["Random\n(kontrol)", "Rollout", "Last-layer\nmentah", "RISE"]
    colors = [COLOR_BASELINE, COLOR_TEXT_LORA, "#3a6fa8", COLOR_VISION_LORA]

    del_auc = [auc[m]["deletion"]["token_f1_auc"] for m in methods]
    ins_auc = [auc[m]["insertion"]["token_f1_auc"] for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))

    ax = axes[0]
    bars = ax.bar(method_labels, del_auc, color=colors)
    ax.set_ylabel("Deletion AUC (token-F1, makin rendah makin baik)")
    ax.set_title("Uji Deletion")
    for b, val in zip(bars, del_auc):
        ax.annotate(f"{val:.4f}", (b.get_x() + b.get_width() / 2, val),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    bars = ax.bar(method_labels, ins_auc, color=colors)
    ax.set_ylabel("Insertion AUC (token-F1, makin tinggi makin baik)")
    ax.set_title("Uji Insertion")
    for b, val in zip(bars, ins_auc):
        ax.annotate(f"{val:.4f}", (b.get_x() + b.get_width() / 2, val),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    # annotate significance vs random directly from file (rise deletion should be p=0.027)
    rise_del_p = sig["rise"]["deletion"]["paired_bootstrap_attention_minus_random"]["two_sided_p_value_approx"]
    rollout_del_p = sig["rollout"]["deletion"]["paired_bootstrap_attention_minus_random"]["two_sided_p_value_approx"]
    fig.text(0.5, -0.02,
              f"Signifikansi vs random (paired bootstrap, deletion): "
              f"RISE p={rise_del_p:.3f} (signifikan), Rollout p={rollout_del_p:.3f} (tidak signifikan)",
              ha="center", fontsize=8, style="italic")

    fig.suptitle(f"Perbandingan 4 metode xAI (n={d['n_examples']} contoh kurasi)", y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig06_xai_faithfulness_comparison.png"))
    fig.savefig(os.path.join(OUT_DIR, "fig06_xai_faithfulness_comparison.pdf"))
    plt.close(fig)
    print(f"[fig06] saved -- RISE deletion p={rise_del_p}, rollout deletion p={rollout_del_p}")


if __name__ == "__main__":
    fig_training_convergence()
    fig_main_results()
    fig_bias_diagnostic()
    fig_efficiency_comparison()
    fig_internvl3_vs_blip()
    fig_xai_faithfulness()
    print(f"\n[done] all figures saved to {OUT_DIR}")
