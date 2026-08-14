# Figure untuk Paper — Index Lengkap

Semua gambar di folder ini dihasilkan dari data eksperimen nyata (bukan mockup/ilustrasi buatan). Enam gambar pertama (`fig01`-`fig06`) baru digenerate lewat `scripts/make_paper_figures.py` (bisa dijalankan ulang kapan saja untuk reproduksi), sisanya (`fig07`-`fig10`) sudah ada dari eksperimen sebelumnya dan tinggal disalin ke sini.

Format: tiap gambar tersedia `.png` (300dpi, untuk draft/preview) dan `.pdf` (vektor, lebih baik untuk print/submission final) — kecuali fig07/fig09/fig10 yang cuma PNG (sumber aslinya begitu).

---

## fig01_training_convergence

**Posisi di paper**: Bagian Metodologi/Eksperimen, dekat penjelasan setup training BLIP (3 fase, 30 epoch).

**Penjelasan singkat untuk caption**: Kurva validation loss BLIP sepanjang 30 epoch training, terbagi 3 fase (LoRA teks epoch 1-10, lanjutan epoch 11-20, + LoRA vision epoch 21-30). Menunjukkan model benar-benar konvergen (val_loss mendatar di akhir fase 3, 1,4312) sebelum dianggap sebagai model final.

**Sumber data**: `logs/full_run_v1_epochs.jsonl`, `full_run_v2_epochs.jsonl`, `full_run_v3_vision_epochs.jsonl`.

---

## fig02_main_results_comparison

**Posisi di paper**: Bagian Hasil Utama (§5.2 di `paper_draft.md`) — figure pendamping tabel utama.

**Penjelasan singkat untuk caption**: Perbandingan CIDEr, BLEU-4, ROUGE-L, dan BERTScore F1 antara baseline zero-shot, LoRA teks-saja, dan model final (LoRA teks+vision), dievaluasi di 2.362 data uji. Model final unggul di semua metrik.

**Sumber data**: `results/metrics.json`.

---

## fig03_bias_diagnostic

**Posisi di paper**: Bagian Diagnosis Bias (§5.3) — bukti visual bahwa LoRA vision memperbaiki dua masalah spesifik.

**Penjelasan singkat untuk caption**: Dua panel: (kiri) akurasi klasifikasi jenis bencana naik dari 58,8% ke 72,3% setelah LoRA ditambahkan ke vision encoder; (kanan) rate prediksi "tidak ada kerusakan" turun dari 83,0% ke 78,5% (garis putus-putus = rate referensi asli 34,2%, menunjukkan bias belum sepenuhnya hilang — dipakai untuk mendukung diskusi limitasi).

**Sumber data**: `results/vision_vs_baseline.json`.

---

## fig04_internvl3_efficiency

**Posisi di paper**: Bagian Studi Banding Backbone Alternatif (§6.3) — **ini figure paling kuat secara naratif**, tunjukkan sebagai figure utama bagian InternVL3.

**Penjelasan singkat untuk caption**: (kiri) Waktu wall-clock per epoch: BLIP ~9,1 menit vs InternVL3 ~142,5 menit (~15,7× lebih lambat). (kanan) Dalam anggaran waktu total yang hampir sama (~4,5 jam), BLIP menyelesaikan 30 epoch sampai konvergen sementara InternVL3 baru 2 epoch dan belum konvergen — ilustrasi trade-off efisiensi vs skala model.

**Sumber data**: `logs/full_run_v{1,2,3_vision}_epochs.jsonl`, `logs/internvl3_run1_epochs.jsonl`, `logs/internvl3_run1_ep2_epochs.jsonl`.

---

## fig05_internvl3_vs_blip_results

**Posisi di paper**: Bagian §6.2, sebagai figure pendamping tabel perbandingan InternVL3 epoch1/epoch2/BLIP.

**Penjelasan singkat untuk caption**: CIDEr, BLEU-4, ROUGE-L, BERTScore F1 untuk InternVL3 epoch 1, epoch 2, dan BLIP final di 2.362 data uji yang sama. Anotasi di pojok kiri atas menunjukkan selisih CIDEr epoch2-vs-BLIP tidak signifikan secara statistik (p=0,081) — **penting dicantumkan supaya pembaca tidak salah baca bar chart sebagai "InternVL3 menang telak"**.

**Sumber data**: `results/internvl3_epoch1_vs_epoch2_vs_blip.json`.

---

## fig06_xai_faithfulness_comparison

**Posisi di paper**: Bagian Explainability (§7.2) — figure ringkasan kuantitatif 4 metode.

**Penjelasan singkat untuk caption**: Deletion AUC dan Insertion AUC untuk 4 metode (Random/kontrol, Rollout, Last-layer mentah, RISE), n=12 contoh kurasi. RISE satu-satunya metode dengan hasil deletion signifikan vs random (p=0,027, dicantumkan di bawah figure) — Rollout tidak signifikan (p=0,269). Sumbu y sengaja dimulai dari 0 (bukan di-zoom) supaya perbedaan yang memang kecil secara absolut tidak terkesan dibesar-besarkan.

**Sumber data**: `results/xai_rise/deletion_insertion_test.json` (field `auc` dan `significance_paired_bootstrap`).

---

## fig07a / fig07b_contoh_komposit

**Posisi di paper**: Bagian Metodologi (§4.3) — figure ilustrasi format input model (bukan hasil eksperimen, tapi penjelasan cara kerja).

**Penjelasan singkat untuk caption**: Contoh citra komposit pre/post-bencana side-by-side (kiri=pra-bencana, kanan=pasca-bencana) yang jadi input model — fig07a untuk sensor Optical, fig07b untuk sensor SAR (menunjukkan pipeline menangani dua jenis sensor citra).

**Sumber data**: dibuat saat Fase 3 (`docs/design_decisions.md` §3), file asli di `docs/figures/composite_*.png`.

---

## fig08_qualitative_rise_overlay

**Posisi di paper**: Bagian Explainability (§7.2), figure utama — grid 4-kolom kualitatif.

**Penjelasan singkat untuk caption**: 4 contoh kualitatif dengan heatmap RISE (di-agregasi khusus token section BANGUNAN) di-overlay pada citra komposit, dibandingkan dengan teks ground truth vs prediksi. Kolom (b) "Akurat" **tidak punya contoh yang benar-benar cocok** di antara 12 data kurasi (dicatat eksplisit di figure) — seluruh contoh dengan kerusakan nyata di ground truth justru diprediksi "tidak ada kerusakan" oleh model, jadi kolom ini pakai contoh terdekat dengan catatan jujur, bukan dipaksakan. Disclaimer signifikansi (p=0,027 deletion, tidak signifikan di insertion, n=12) tercantum di bawah figure — **jangan dihapus saat resize/reformat**, itu bagian penting dari kejujuran metodologis figure ini.

**Sumber data**: `docs/figures/qualitative_comparison_rise_selection.json` (metadata seleksi 4 contoh), dihitung dari `scripts/make_qualitative_figure_rise.py`.

---

## fig09_rollout_deletion_insertion_curves

**Posisi di paper**: Bagian Explainability (§7.1), pendamping temuan negatif rollout — opsional, bisa masuk lampiran kalau ruang terbatas.

**Penjelasan singkat untuk caption**: Kurva deletion/insertion (token-F1 dan CIDEr) untuk metode rollout dibanding random, n=12. Kurva yang saling berhimpit dengan random adalah bukti visual bahwa rollout tidak faithful — melengkapi angka AUC di fig06.

**Sumber data**: `results/xai/deletion_insertion_test.json`.

---

## fig10_rise_deletion_insertion_curves

**Posisi di paper**: Bagian Explainability (§7.2), pendamping fig09 sebagai perbandingan langsung "sebelum vs sesudah".

**Penjelasan singkat untuk caption**: Kurva deletion/insertion yang sama untuk RISE — dibandingkan dengan fig09, kurva deletion RISE terlihat lebih terpisah dari random, konsisten dengan hasil signifikansi p=0,027.

**Sumber data**: `results/xai_rise/deletion_insertion_test.json`.

---

## Rekomendasi urutan pemakaian (kalau paper punya batasan jumlah figure)

**Wajib (5 figure inti, cerita lengkap tersampaikan)**: fig02, fig03, fig04, fig06, fig08.
**Kuat tapi opsional**: fig01 (training curve — umum di paper ML, tapi tidak selalu wajib), fig05 (bisa digabung jadi 1 kalimat di tabel teks kalau ruang mepet), fig07a (1 contoh komposit sudah cukup untuk ilustrasi metodologi).
**Lampiran/opsional**: fig09, fig10 (kurva detail — AUC ringkasannya sudah di fig06), fig07b (redundan dengan fig07a kecuali ingin eksplisit tunjukkan SAR).

## Reproduksi

```bash
python scripts/make_paper_figures.py          # fig01-fig06
python scripts/make_qualitative_figure_rise.py # fig08 (fig07/09/10 sudah ada dari eksperimen sebelumnya)
```
