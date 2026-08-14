# IndoBLIP-Post-Disaster — Ringkasan untuk Diskusi dengan Dosen Pembimbing

**Tujuan dokumen ini:** bahan diskusi sebelum submit paper GEMASTIK — apa yang sudah selesai (dengan bukti nyata), apa yang sedang dieksplorasi, dan apa yang masih jadi keputusan terbuka. Semua angka di bawah adalah hasil pengukuran nyata (bukan estimasi), dengan pointer ke file sumbernya masing-masing supaya bisa diverifikasi ulang kapan saja.

---

## 1. SUDAH DILAKUKAN (selesai, terverifikasi, siap dilaporkan)

### 1.1 Audit dataset (Fase 0)
Dataset: DisasterM3 (Kingdrone-Junjue/DisasterM3, NeurIPS 2025). Struktur dan jumlah data asli sudah diaudit langsung dari file mentah (bukan dari klaim paper), dan **beberapa klaim draft paper awal tidak cocok dengan data yang benar-benar dirilis**:

| Klaim | Draft paper | Data real (terukur) |
|---|---|---|
| Pasangan bi-temporal | 26.988 | **12.861** (~47-54%) |
| Train/test captioning | 17.190 / 5.024 | **7.766 / 2.363** (~45-47%) |
| Total instruksi | 123.010 | 123.010 ✅ cocok |

→ Sumber: `docs/dataset_audit.md`, `docs/draft_vs_real_comparison.md`. **Ini bukan salah kami** — dua metode pengukuran independen (dari JSON dan dari isi zip mentah) sepakat, jadi kemungkinan besar rilis publik dataset memang lebih kecil dari yang dideskripsikan di paper asli DisasterM3.

### 1.2 Translasi dataset (Fase 1)
Caption asli 100% Bahasa Inggris (diverifikasi, 0 kata Indonesia ditemukan di 282.271 field teks). Ditranslate pakai NLLB-200-distilled-600M, section-aware (per bagian laporan), dengan QC dan perbaikan khusus section KESIMPULAN yang awalnya sering terpotong. → `docs/translation_report.md`.

### 1.3 Arsitektur & fine-tuning (Fase 2-3)
- **Model dasar**: BLIP (`Salesforce/blip-image-captioning-base`), 224M parameter (ViT-B/16 86M + BERT-base decoder 138M).
- **Adaptasi tokenizer ke Bahasa Indonesia** (`indolem/indobert-base-uncased`) — keputusan krusial, karena tokenizer BLIP asli (WordPiece Inggris) memecah kata Indonesia jadi 2,64 token/kata vs 1,20 token/kata setelah adaptasi.
- **LoRA** pada text decoder, lalu diperluas ke vision encoder (lihat 1.5).
- Pasangan citra pre/post digabung jadi 1 komposit side-by-side 384×384 (bukan modifikasi arsitektur).
- → `docs/design_decisions.md` §0-§5 (semua keputusan desain didokumentasikan dengan alasannya).

### 1.4 Evaluasi (Fase 4) — hasil akhir real
| Metrik | Baseline zero-shot | LoRA teks-saja | **LoRA teks+vision (final)** |
|---|---|---|---|
| CIDEr | 0,0001 | 0,0980 | **0,1202** |
| BLEU-4 | 0,0000 | 0,2044 | **0,2095** |
| ROUGE-L | 0,0185 | 0,4103 | **0,4156** |
| BERTScore F1 | 0,3470 | 0,7943 | **0,7971** |

Signifikansi (paired bootstrap, 10.000 resample, per-example CIDEr):
- Final vs baseline: diff +0,1201, CI95 [0,1078; 0,1329] — signifikan.
- Final vs ablasi teks-saja: diff +0,0221, CI95 [0,0099; 0,0345], p=0,0004 — signifikan.

→ `results/metrics_table.md`, `results/metrics.json`.

### 1.5 Diagnosis bias & perbaikan vision-LoRA (Fase 5)
Ditemukan 2 bias di model awal (LoRA teks-saja): akurasi jenis bencana rendah, dan model terlalu sering bilang "tidak ada kerusakan". Root cause diinvestigasi secara disiplin:
- Kontrol murah dulu (beam search vs greedy) → **terbukti BUKAN penyebab** (beam malah lebih buruk).
- Baru lanjut ke eksperimen mahal: buka vision encoder lewat LoRA (bukan dibekukan total) → **terbukti membantu nyata**:

| Diagnostik | LoRA teks-saja | **+ vision LoRA** | Rate asli di data |
|---|---|---|---|
| Akurasi jenis bencana (BENCANA) | 58,8% | **72,3%** | — |
| Rate "tidak ada kerusakan" (BANGUNAN) | 83,0% | **78,5%** | 34,2% |

**Catatan jujur**: bias belum hilang total (78,5% vs 34,2% masih jauh) — ini limitasi yang didokumentasikan apa adanya, bukan disembunyikan. → `docs/design_decisions.md` §5.

### 1.6 Explainability / xAI (Fase 6)
Cross-attention rollout diimplementasikan dan diuji faithfulness-nya (bukan cuma dibuat lalu diasumsikan benar). Temuan-temuan **negatif yang jujur dilaporkan** (bukan angka bagus yang dicari-cari):
- Peta rollout hampir identik dengan raw last-layer attention (r=0,9998) — rollout multi-layer tidak banyak menambah dibanding versi paling sederhana pada arsitektur ini.
- Peta attention **token-invariant** — token yang berbeda dalam satu caption menghasilkan peta atensi yang hampir sama (temuan penting: attention di arsitektur ini kurang bisa dipercaya sebagai penjelasan per-kata).
- Uji occlusion: menutup separuh gambar post-disaster **tidak** mengubah output sebanyak yang diharapkan (model toleran sampai 90% patch dihapus) — mengindikasikan sinyal visual yang dipelajari cenderung global/kasar, bukan grounding lokal yang presisi.

→ `docs/design_decisions.md` §6, `results/xai/`, `results/xai_examples/`, `notebooks/06_xai_attention_rollout.ipynb`.

### 1.7 Dokumentasi & deliverable pendukung
- **6 notebook** (`notebooks/01-06`) — EDA sampai xAI, semua benar-benar dieksekusi dengan output nyata, bisa dibuka langsung tanpa rerun.
- **Demo Gradio** (`demo/app.py`) — upload pasangan gambar pre/post → generate laporan, sudah diuji jalan nyata.
- **README.md**, `requirements.txt`, repo GitHub sudah rapi (histori bersih, dataset/checkpoint besar di-exclude dengan benar).
- **`docs/draft_vs_real_comparison.md`** — perbandingan eksplisit klaim draft paper (fiktif/mock-up) vs hasil real kita, supaya tidak ada yang salah kutip di paper.

### 1.8 Verifikasi checkpoint eksternal
Sempat diminta coba checkpoint BLIP hasil fine-tune orang lain (`indoblip-lr-5e-06-epoch-30`, dari proyek skripsi terpisah, domain captioning umum bukan bencana). Diuji langsung dengan data real: **gagal total** di tugas kita (CIDEr 0,0000, output berupa token rusak berulang) — karena domain berbeda, tokenizer Inggris tidak diadaptasi, dan bug `eos_token_id` yang sama seperti yang sudah kami temukan & perbaiki sendiri. Kesimpulan: model kami sendiri tetap yang terbaik untuk tugas ini.

---

## 2. SEDANG DIEKSPLORASI (preliminary, belum diimplementasikan)

### 2.1 Kandidat backbone alternatif: Qwen2-VL-2B-Instruct
Riset (deep research, sitasi arXiv terverifikasi) merekomendasikan **Qwen2-VL-2B-Instruct** sebagai kandidat backbone alternatif/pembanding terkuat:
- Tokenizer BPE 151k (jauh lebih ramah Bahasa Indonesia daripada WordPiece BLIP).
- Dukungan multi-image native → tidak perlu trik komposit side-by-side seperti sekarang.
- Lisensi Apache 2.0.
- **Preseden domain langsung**: pembuat dataset DisasterM3 sendiri sudah fine-tune Qwen2.5-VL dan InternVL3 di dataset yang sama persis.
- Sudah diverifikasi nyata di hardware kita: **QLoRA 4-bit (bitsandbytes) benar-benar jalan di GPU RTX 5050** (diuji langsung, forward+backward pass berhasil) — jadi kelayakan teknis untuk fine-tune di 8GB VRAM tidak lagi jadi asumsi, sudah dibuktikan.
- Kandidat lain yang dipertimbangkan tapi diberi peringkat lebih rendah: InternVL2-2B/1B (lisensi MIT paling bersih), PaliGemma 2-3B (tokenizer Gemma 256k bagus, tapi LLM-nya condong Inggris).
- Florence-2 dan SmolVLM2 **didiskualifikasi** sebagai kandidat utama — keduanya tidak mendukung Bahasa Indonesia dengan baik (akan mengulang masalah fragmentasi tokenizer yang sama seperti BLIP).

**Status**: baru riset literatur + 1 verifikasi teknis (bitsandbytes di GPU kita). Belum ada eksperimen fine-tuning nyata dengan Qwen2-VL.

### 2.2 Ide penambahan jumlah objek (object count) ke laporan
Muncul dari pertanyaan: bisakah laporan kerusakan menyebutkan jumlah bangunan/objek untuk menambah kredibilitas?

**Temuan investigasi data (nyata, dari file JSON mentah)**:
- DisasterM3 punya task terpisah `Building Damage Counting` dan `Road Damage Counting`, dengan overlap pasangan gambar 80-95% terhadap task captioning yang kita pakai — secara data, feasible untuk digabung.
- **Tapi**: ground truth `Building Damage Counting` ternyata berbentuk **jawaban pilihan ganda** (MCQ, 5 opsi), bukan hasil hitung dari anotasi per-bangunan yang bisa kita verifikasi sendiri. Tidak ada mask/box per-instance bangunan yang diekspos di dataset.
- `Road Damage Counting` namanya menyesatkan — isinya **persentase area** (mis. "5,22%"), bukan hitungan objek.

**Kesimpulan sementara (penting untuk didiskusikan)**: menempelkan angka ke caption hasil generate teks **berisiko menurunkan kredibilitas**, bukan menambah — karena model captioning generatif (seperti BLIP, atau VLM manapun tanpa mekanisme counting eksplisit) tidak benar-benar menghitung objek dari piksel, hanya menebak angka yang "terlihat masuk akal" secara pola. Ini bukan spekulasi kami saja — **paper DisasterM3 sendiri secara eksplisit menyebut "damage object counting insensitivity" sebagai kelemahan yang masih dialami model-model besar sekalipun (Qwen2.5-VL, InternVL3, dll)**.

**Arah yang lebih menjanjikan (disukai, jadi ide untuk dikembangkan)**: arsitektur **multi-head/multi-task** — head terpisah yang dilatih khusus untuk task counting (klasifikasi/regresi, sama seperti cara DisasterM3 sendiri mengevaluasi task ini), bukan meminta decoder teks men-generate angka. Hasil counting head lalu digabung ke laporan lewat post-processing, bukan lewat generation langsung.

**Status**: ide + investigasi kelayakan data. Belum ada desain arsitektur atau implementasi.

---

## 3. AKAN DILAKUKAN / KEPUTUSAN TERBUKA (bahan diskusi dengan dosen)

Ini yang paling relevan untuk dibahas bareng dosen, karena melibatkan trade-off waktu vs cakupan sebelum deadline submit:

1. **Prioritas sisa waktu**: lanjut ke eksperimen Qwen2-VL (backbone alternatif), atau ke arsitektur multi-head counting, atau fokus memoles laporan hasil yang sudah ada untuk submission? Ketiganya legitimate tapi butuh waktu berbeda-beda dan tidak realistis dikerjakan semua sebelum deadline.
2. **Bagaimana melaporkan limitasi**: bias residual (no-damage over-prediction 78,5% vs 34,2%) dan temuan xAI yang negatif (attention kurang faithful, token-invariant) — apakah dilaporkan apa adanya sebagai kontribusi ilmiah yang jujur (rekomendasi kami), atau dosen punya pertimbangan lain soal framing untuk kompetisi?
3. **Cakupan Fase 7 (laporan akhir)**: sebagian besar sudah tercakup lewat notebook + README + docs — perlu dikonfirmasi apakah masih ada format/struktur laporan akhir spesifik yang diwajibkan panitia GEMASTIK yang belum kita penuhi.
4. **Soal dataset yang mismatch dari draft paper** (bi-temporal pairs, ukuran split) — apakah perlu disebutkan eksplisit di paper sebagai catatan/limitasi, atau cukup dilaporkan angka real tanpa membahas selisihnya secara eksplisit.
5. **Skip ablasi matched-epoch** (kontrol yang sengaja dilewati saat memutuskan lanjut ke vision-LoRA, demi efisiensi waktu) — didokumentasikan sebagai trade-off metodologis; perlu dicek apakah dosen menganggap ini cukup atau perlu ditambal sebelum submit.

---

## 4. Potensi Arah Diskusi Lain (lebih general)

Poin-poin di §3 sengaja spesifik karena sudah ada bukti/investigasi di baliknya. Bagian ini sebaliknya — **sengaja dibuat umum**, sebagai pintu masuk diskusi yang lebih terbuka, supaya dosen bisa mengarahkan ke aspek yang beliau anggap penting tanpa kita harus punya jawaban detail siap pakai untuk semuanya. Bisa dipilih sebagian saja sesuai arah yang paling relevan menurut dosen.

1. **Posisi/framing paper ini secara keilmuan** — apakah ditonjolkan sebagai kontribusi rekayasa sistem (pipeline end-to-end untuk domain baru), studi empiris (analisis bias & keterbatasan model vision-language pada domain bencana), atau kontribusi metodologis (adaptasi tokenizer/LoRA untuk bahasa low-resource)? Framing ini akan mempengaruhi bagian mana yang perlu diperdalam sebelum submit.

2. **Generalisasi ke luar proyek ini** — apakah pendekatan (adaptasi tokenizer, LoRA vision+teks, evaluasi bias) punya nilai yang lebih luas di luar dataset DisasterM3 atau Bahasa Indonesia saja — misalnya untuk bahasa daerah/low-resource lain, atau domain citra lain di luar bencana.

3. **Sisi etis dan risiko praktis** — mengingat model masih over-predict "tidak ada kerusakan", apa implikasinya kalau sistem semacam ini dipakai sungguhan dalam respons bencana nyata (risiko false-negative pada kerusakan asli). Ini bisa jadi bagian diskusi/limitasi yang berbobot, bukan sekadar angka teknis.

4. **Constraint sumber daya sebagai sudut pandang kontribusi** — seluruh eksperimen dikerjakan di 1 GPU laptop 8GB, bukan cluster riset besar. Ini bisa dibingkai sebagai nilai tambah (aksesibilitas/reproducibility di sumber daya terbatas), tergantung apakah dosen melihat ini relevan untuk ditonjolkan.

5. **Metodologi evaluasi untuk generative task terstruktur** — diskusi terbuka soal apakah metrik captioning standar (CIDEr/BLEU/ROUGE/BERTScore) benar-benar representatif untuk laporan multi-section seperti ini, atau perlu pendekatan evaluasi lain (mis. evaluasi per-section, human evaluation, dsb) — tanpa harus komit ke satu metrik pengganti tertentu dulu.

6. **Explainability sebagai arah riset, bukan cuma hasil satu eksperimen** — temuan bahwa attention rollout kurang faithful bisa dibuka jadi diskusi lebih luas soal keterbatasan interpretability method berbasis attention pada arsitektur vision-language secara umum, bukan cuma spesifik ke BLIP kita.

7. **Praktik audit data sebagai bagian dari kontribusi metodologis** — proses memverifikasi ulang klaim dataset/paper alih-alih menerima begitu saja bisa didiskusikan sebagai bagian dari kontribusi reproducibility, terlepas dari hasil spesifik mismatch yang ditemukan.

8. **Arah pengembangan lanjutan secara umum** — di luar dua ide spesifik di §2 (Qwen2-VL, multi-head counting), bisa dibuka diskusi lebih luas soal kategori pengembangan apa yang dosen anggap paling bernilai: backbone lebih besar/baru, multi-task learning, human-in-the-loop verification, atau arah lain yang belum terpikirkan tim.

---

*Dokumen ini dibuat sebagai bahan diskusi berkelanjutan — akan diperbarui seiring progres. Semua klaim di atas bisa ditelusuri ke file sumber aslinya di repo (`docs/`, `results/`, `notebooks/`).*
