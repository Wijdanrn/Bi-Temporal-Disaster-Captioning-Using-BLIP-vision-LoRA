# Eksperimen InternVL3-1B-hf — Kandidat Backbone Alternatif (Dari Awal Sampai Akhir)

**Tujuan dokumen ini:** catatan lengkap dan jujur tentang eksperimen mencoba backbone VLM alternatif (InternVL3-1B-hf) sebagai pembanding terhadap model utama kami (BLIP + LoRA). Ditulis supaya siapa pun di tim bisa memahami apa yang terjadi, kenapa keputusan tertentu diambil, dan bagaimana hasilnya bisa (atau tidak bisa) dipakai di paper — tanpa harus menelusuri ulang percakapan panjangnya.

Semua angka di dokumen ini adalah hasil pengukuran nyata dari log/file yang ada di repo (`logs/`, `results/`, `checkpoints/`) — bisa ditelusuri ulang kapan saja.

---

## 1. Kenapa eksperimen ini dilakukan

Model utama kami (BLIP, 224M parameter, tokenizer diadaptasi manual ke Bahasa Indonesia) sudah selesai dan hasilnya solid (lihat `results/metrics_table.md`). Tapi muncul dua pertanyaan yang perlu dijawab dengan bukti, bukan asumsi:

1. **Riset literatur** (`Deep Research/compass_artifact_...md`) merekomendasikan backbone VLM modern (Qwen2-VL, InternVL) sebagai kandidat lebih kuat untuk Bahasa Indonesia, karena tokenizer-nya (BPE ~150k token) jauh lebih ramah bahasa non-Inggris dibanding tokenizer BLIP asli.
2. Saat membandingkan BLIP (224M) dengan model besar seperti Qwen2.5-VL-3B (yang dilatih terpisah oleh rekan tim), muncul kekhawatiran valid: **apakah perbandingan itu apple-to-apple?** 224M vs 3B itu beda skala besar. Perlu ada titik tengah (~1B) untuk membuat cerita perbandingan lebih adil dan bertahap.

**Kandidat yang dipilih: `OpenGVLab/InternVL3-1B-hf`** (~0,94B parameter) — dipilih dari beberapa alternatif karena:
- Ukuran paling dekat ke "1B" dari semua kandidat yang dicek.
- Sudah di-port native ke `transformers` (kelas `InternVLForConditionalGeneration`) — **tidak perlu `trust_remote_code=True`**, beda dengan `OpenGVLab/InternVL2-1B` versi asli yang masih pakai kode custom repo (risiko lebih tinggi).
- Tokenizer backbone Qwen2 (BPE 151.674 token) — sama keluarga dengan rekomendasi utama riset.
- Mendukung **multi-image native** — pasangan gambar pre/post bisa dikirim sebagai 2 gambar terpisah, tidak perlu trik komposit side-by-side seperti BLIP.

---

## 2. Rintangan teknis yang dihadapi (dan cara mengatasinya)

Bagian ini penting untuk reproducibility — banyak dari masalah ini bukan soal desain, tapi soal infrastruktur, dan solusinya sudah terbukti jalan.

### 2.1 Unsloth merusak environment
Sempat dicoba menginstal `unsloth` (library percepatan training yang dipakai rekan tim untuk Qwen2.5-VL-3B) untuk mempercepat training. Ternyata **`pip install unsloth` diam-diam mengganti build PyTorch kami** (nightly `cu128` khusus untuk GPU Blackwell RTX 5050) menjadi build CPU-only, dan bahkan setelah diperbaiki, `torchvision` ikut tidak sinkron sehingga `peft`/`transformers` gagal di-import.

**Solusi**: instal ulang `torch` dan `torchvision` dari index nightly `cu128` yang sama, diverifikasi ulang end-to-end (CUDA, bitsandbytes, peft, transformers, forward pass model) sebelum lanjut. **Keputusan**: tidak mencoba Unsloth lagi — Unsloth juga tidak mendukung arsitektur InternVL sama sekali (cuma Qwen-VL, Llama Vision, Gemma 3, Pixtral/Llava), jadi tidak relevan untuk kandidat ini.

### 2.2 Download besar terpotong di jaringan lokal
Download model dari HuggingFace lewat `snapshot_download`/`hf_hub_download` standar **selalu terpotong di ~1KB** untuk file besar (`tokenizer.json` 11MB, `model.safetensors` 1,87GB) — pola yang identik dengan masalah SSL/proxy yang pernah dialami saat download dataset DisasterM3 di awal proyek.

**Solusi**: dibuat ulang downloader byte-range chunked (`scripts/chunked_hf_download.py`), pakai 128 chunk kecil (~14,6MB/chunk) dengan resume-per-chunk (bukan restart total saat retry) dan concurrency dibatasi 6 — berhasil 100% dengan verifikasi ukuran file exact.

### 2.3 OOM saat training
Percobaan awal (`batch_size=4`, tanpa gradient checkpointing) gagal dengan `CUDA error: out of memory` — sempat muncul sebagai error cuBLAS misterius dulu sebelum akhirnya jelas OOM asli. Memori sempat "terlihat" jalan di atas 11GB (melebihi VRAM fisik 8,15GB) karena Windows diam-diam memakai shared system memory sebagai overflow (lambat, dan akhirnya tetap OOM).

**Solusi**: aktifkan `gradient_checkpointing` + `model.enable_input_require_grads()` (wajib dipasangkan saat pakai PEFT), turunkan ke `batch_size=1` dengan `grad_accum=4`. Memori jadi stabil di ~4,3-4,7GB.

### 2.4 Label masking salah total (bug diam-diam)
Dataset pipeline awal (2 gambar × dynamic image tiling InternVL) menghasilkan prompt sepanjang **5.186 token** — jauh melebihi `max_length=1024` yang di-set, sehingga urutan yang ter-truncate **seluruhnya masuk ke wilayah prompt**, dan label training (`labels[:prompt_len] = -100`) **ter-mask 100%** — training akan berjalan tanpa sinyal sama sekali kalau tidak ketahuan.

**Solusi**: set `max_patches=1` di image processor (menonaktifkan dynamic tiling InternVL, 1 tile per gambar saja — prinsip sama seperti resize tetap tanpa tiling yang dipakai BLIP), memangkas panjang prompt dari 5.186 → 578 token.

### 2.5 Inferensi lambat, lalu ditemukan solusi batching
Generate satu-per-satu awalnya **33,3 detik/contoh** — untuk 2.362 test set itu ~21,8 jam, tidak realistis. Setelah dicoba batching (generate banyak sekaligus dengan left-padding, wajib untuk causal-LM batched generation):
- batch=8: 4,43 detik/contoh (7,5x lebih cepat)
- **batch=32: 1,19-1,3 detik/contoh (≈28x lebih cepat dari sequential)**, memori cuma ~4,1GB

Full test set (2.362) yang tadinya ~22 jam jadi **~51 menit**.

---

## 3. Setup training

- **Model**: `OpenGVLab/InternVL3-1B-hf`, 938.193.024 parameter total.
- **LoRA**: r=8, alpha=32, dropout=0.1, target `q_proj`/`v_proj` — diterapkan ke **LLM dan vision tower sekaligus** (menerapkan langsung pelajaran dari BLIP bahwa vision encoder tidak boleh dibekukan total, tanpa perlu re-discover dari nol). Trainable: 1.327.104 parameter (0,14% dari total).
- **Tidak ada operasi tokenizer/embedding surgery** seperti di BLIP — tokenizer Qwen2 InternVL3 sudah punya cakupan Bahasa Indonesia yang baik dari awal.
- **Data**: sama persis dengan BLIP — `data/processed/captions_train.jsonl` (6.999) / `captions_val.jsonl` (767), tapi gambar pre/post dikirim sebagai **2 gambar terpisah** (bukan komposit side-by-side).
- **Prompt**: instruksi Bahasa Indonesia langsung (`"Jelaskan situasi kerusakan secara komprehensif berdasarkan gambar sebelum dan sesudah bencana..."`) — beda dari BLIP yang tidak memakai instruksi teks sama sekali (captioning murni).
- Script: `scripts/train_internvl3.py`, `scripts/internvl3_dataset.py`, `scripts/generate_predictions_internvl3.py`.

---

## 4. Hasil real, tahap demi tahap

### 4.1 Epoch 1
- val_loss: **0,8709**, waktu: 8.473,7 detik (2 jam 21 menit)
- Checkpoint: `checkpoints/internvl3_run1/best`
- Dievaluasi di 2.362 test set penuh → **kalah di semua metrik** dari BLIP final (wajar, baru 1 epoch vs 30 epoch BLIP)

### 4.2 Epoch 2 (resume dari checkpoint epoch 1, bukan dari nol)
- val_loss: **0,8055** (turun signifikan dari 0,8709)
- waktu: 8.631,8 detik (2 jam 24 menit)
- Checkpoint: `checkpoints/internvl3_run1_ep2/best`

### 4.3 Perbandingan 3-arah di 2.362 test set (identik untuk semua model)

| Metrik | InternVL3 epoch 1 | **InternVL3 epoch 2** | BLIP final (30 epoch) |
|---|---|---|---|
| CIDEr | 0,0846 | **0,1327** | 0,1202 |
| BLEU-4 | 0,1950 | **0,2134** | 0,2095 |
| ROUGE-L | 0,4081 | 0,4149 | 0,4156 |
| BERTScore F1 | 0,7903 | 0,7954 | 0,7971 |
| Akurasi BENCANA | 43,3% | 53,1% | **72,3%** |
| Rate "tidak ada kerusakan" | 82,6% | 82,3% | 78,5% (ref: 34,2%) |
| Caption unik | 76,6% | **96,9%** | 86,5% |

**Signifikansi statistik** (paired bootstrap, 10.000 resample, CIDEr per-contoh):
- Epoch 2 vs epoch 1: selisih **+0,0481**, CI95 [0,0366; 0,0594], **p≈0,0000** — peningkatan nyata dan signifikan.
- Epoch 2 vs BLIP final: selisih **+0,0126**, CI95 [-0,0016; 0,0265], **p=0,081** — **belum signifikan** di ambang 0,05 (meski 96% resample bootstrap condong ke InternVL3) → kesimpulan yang tepat adalah **"secara statistik setara"**, bukan "menang" atau "kalah".

Sumber: `results/internvl3_epoch1_vs_epoch2_vs_blip.json` (skrip: `scripts/score_internvl3_epoch2.py`).

### 4.4 Perbandingan di subset 1182 (item yang sama dengan evaluasi Qwen2.5-VL-3B rekan tim)

Supaya nanti bisa langsung disandingkan 3-arah dengan hasil Qwen2.5-VL-3B begitu tersedia, InternVL3 (epoch 1 & 2) dan BLIP juga di-skor ulang khusus di **1182 item yang identik** dengan subset yang dipakai rekan tim (`test_filenames_first_1182.csv` — subset non-random, cuma cover 6/10 tipe bencana, lihat `results/shared_subset_1182_matched_indices.json`). Tidak perlu generate ulang — tinggal filter dari prediksi full-set yang sudah ada.

| Metrik | InternVL3 epoch 1 | **InternVL3 epoch 2** | BLIP final |
|---|---|---|---|
| CIDEr | 0,0801 | **0,1223** | 0,1071 |
| BLEU-4 | 0,1898 | **0,2073** | 0,2056 |
| ROUGE-L | 0,4025 | 0,4079 | 0,4112 |
| BERTScore F1 | 0,7877 | 0,7924 | 0,7949 |
| Akurasi BENCANA | 26,8% | 36,3% | **59,9%** |
| Caption unik | 81,7% | **98,5%** | 92,2% |

Signifikansi: epoch2 vs BLIP CIDEr diff +0,0152, p=0,101 (belum signifikan, konsisten dengan hasil full-set). Polanya sama seperti full-set: CIDEr/BLEU-4 setara, tapi BENCANA accuracy InternVL3 masih jauh tertinggal.

Sumber: `results/shared_subset_1182_internvl3_vs_blip.json` (skrip: `scripts/score_shared_subset_internvl3.py`).

---

## 5. Narasi efisiensi (waktu training vs BLIP)

Ini temuan yang paling kuat secara naratif untuk paper, karena datanya sangat jelas:

| | BLIP | InternVL3 |
|---|---|---|
| Total parameter | 223.971.644 (0,224B) | 938.193.024 (0,938B) — **4,19x lebih besar** |
| Total epoch selesai | **30** (3 fase, 10+10+10) | **2** |
| Total waktu wall-clock | 16.329,3 detik = **4 jam 32 menit** | 17.105,5 detik = **4 jam 45 menit** |
| Waktu per epoch | 544,3 detik ≈ **9,1 menit** | 8.552,8 detik ≈ **142,5 menit (2,4 jam)** |
| Status konvergensi | **Konvergen/plateau** | **Belum konvergen** (masih menurun tajam) |

**Insight utama**: dalam waktu wall-clock yang **hampir sama** (beda cuma 13 menit), BLIP menyelesaikan **30 epoch sampai konvergen**, sementara InternVL3 baru sanggup **2 epoch dan masih jauh dari konvergen** — InternVL3 butuh **~15,7x waktu per-epoch** lebih lama, kemungkinan kombinasi dari: ukuran model 4,2x lebih besar, gradient checkpointing (trade compute demi memori terbatas), sequence length lebih panjang, dan `batch_size=1` yang terpaksa dipakai karena VRAM 8GB.

**Cara membingkai di paper (jujur, terukur, tidak overclaim)**:
> Meskipun InternVL3-1B menunjukkan performa kompetitif hanya dalam 2 epoch — secara statistik setara dengan BLIP yang sudah konvergen penuh pada CIDEr/BLEU-4 — pencapaian ini membutuhkan anggaran waktu wall-clock yang sebanding dengan BLIP mencapai konvergensi penuh 30 epoch. Ini menyoroti trade-off nyata antara kekuatan pretraining backbone besar vs efisiensi adaptasi model kecil: BLIP (224M) mencapai hasil stabil dalam ~4,5 jam di satu GPU laptop, sementara InternVL3 (938M) butuh ~15,7x waktu per-epoch untuk mendekati performa setara, dan belum konvergen dalam anggaran waktu yang sama.

---

## 6. Kesimpulan & rekomendasi penggunaan di paper

1. **Aman dilaporkan sebagai temuan jujur, bukan hasil final InternVL3** — training belum konvergen (loss masih turun tajam di kedua checkpoint). Jangan menyajikan angka epoch-2 sebagai "performa maksimal InternVL3", karena itu tidak diketahui — kita hanya tahu performanya *setelah 2 epoch dengan anggaran waktu tertentu*.
2. **BENCANA accuracy (klasifikasi jenis bencana) adalah kelemahan paling nyata InternVL3** saat ini (53,1% vs 72,3%) — kemungkinan butuh lebih banyak epoch untuk model mempelajari pembedaan tipe bencana secara tajam, berbeda dari kemampuan menulis teks yang mengalir (CIDEr/BLEU) yang tampaknya lebih cepat dipelajari berkat pretraining kuat.
3. **Distinct caption rate InternVL3 jauh lebih tinggi** (96,9% vs 86,5%) — sinyal positif bahwa model tidak jatuh ke mode collapse/jawaban generik berulang, berbeda dari kekhawatiran yang muncul di analisis bias BLIP awal.
4. **Narasi efisiensi (Bagian 5) adalah kontribusi yang paling kuat dan aman digunakan** — datanya terukur otomatis, tidak butuh interpretasi subjektif, dan relevan untuk argumen "aksesibilitas riset dengan sumber daya terbatas" yang sudah jadi salah satu sudut pandang kontribusi proyek ini.
5. **Bukan perbandingan arsitektur yang adil** — beda skala (4,2x parameter), beda jumlah training, beda availability tooling (Unsloth mendukung Qwen2-VL tapi tidak InternVL). Semua ini harus disebutkan eksplisit di paper supaya tidak ada klaim yang terlalu jauh dari bukti.

---

## 7. Semua file sumber yang relevan

| File | Isi |
|---|---|
| `scripts/train_internvl3.py` | Training loop (LoRA, resume-capable) |
| `scripts/internvl3_dataset.py` | Dataset + collator (2-gambar native, prompt Indonesia) |
| `scripts/generate_predictions_internvl3.py` | Generate batched (left-padding) |
| `scripts/chunked_hf_download.py` | Downloader byte-range untuk mengatasi network yang memotong file besar |
| `scripts/score_internvl3_epoch2.py` | Skoring 3-arah (epoch1/epoch2/BLIP) |
| `logs/internvl3_run1_*.jsonl`, `logs/internvl3_run1_ep2_*.jsonl` | Log training real per-step/per-epoch |
| `checkpoints/internvl3_run1/best`, `checkpoints/internvl3_run1_ep2/best` | Checkpoint LoRA (epoch 1 dan epoch 2) |
| `results/predictions_test_internvl3_epoch1.jsonl`, `..._epoch2.jsonl` | Prediksi mentah 2.362 test set |
| `results/internvl3_epoch1_vs_epoch2_vs_blip.json` | Hasil skoring 3-arah full 2.362 test set (sumber tabel Bagian 4.3) |
| `scripts/score_shared_subset_internvl3.py` | Skoring InternVL3+BLIP di 1182 item yang sama dengan evaluasi Qwen |
| `results/shared_subset_1182_internvl3_vs_blip.json` | Hasil skoring di subset 1182 (sumber tabel Bagian 4.4) — siap disandingkan begitu skor Qwen2.5-VL-3B tersedia |
| `results/shared_subset_1182_metrics.json` | Skor BLIP saja (baseline/text-only/vision-LoRA) di subset 1182 — dibuat lebih awal, sebelum InternVL3 ada |
