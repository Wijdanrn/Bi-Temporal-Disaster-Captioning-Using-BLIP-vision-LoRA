# Hasil yang Layak Dipakai di Paper — Index

Folder ini kumpulan salinan hasil-hasil yang sudah terverifikasi dan siap dikutip di paper. **Sumber aslinya tetap di `results/` dan `docs/`** — folder ini cuma kurasi supaya tidak perlu menelusuri seluruh repo. Kalau butuh detail lebih lengkap (skrip, log mentah, dsb), rujuk balik ke file sumber yang disebut di tiap bagian.

Semua angka di sini nyata (dihitung dari data asli, bukan estimasi/karangan) — bisa direproduksi ulang lewat skrip yang disebut.

---

## 1. Dataset (Fase 0)

- **`dataset_audit.md`** — audit dataset DisasterM3, termasuk perbandingan klaim draft paper vs data real (beberapa klaim draft paper tidak cocok dengan data yang benar-benar dirilis — dilaporkan apa adanya).

## 2. Model utama: BLIP + LoRA (hasil final, sudah settled)

- **`blip_final_metrics.json`** / **`blip_final_metrics_table.md`** — hasil evaluasi akhir 3-arah (baseline zero-shot / LoRA teks-saja / LoRA teks+vision) di 2.362 test set penuh. Ini **angka headline utama** proyek.
- **`blip_vision_lora_ablation.json`** — bukti bahwa menambah LoRA ke vision encoder (bukan cuma teks) memperbaiki bias signifikan (akurasi jenis bencana 58,8%→72,3%).
- **`blip_beam_vs_greedy_ablation.json`** — kontrol eksperimen yang membuktikan bias BUKAN disebabkan strategi decoding (beam search malah lebih buruk).

## 3. Explainability / xAI (Fase 6)

- **`xai_summary/section_attention_aggregate.json`** — agregat attention rollout per-section (n=150).
- **`xai_summary/occlusion_test.json`** — uji faithfulness: tutup separuh gambar, lihat perubahan output.
- **`xai_summary/deletion_insertion_test.json`** + **`deletion_insertion_curves.png`** — uji faithfulness kedua (792 generasi).
- **Temuan penting (negatif, dilaporkan jujur)**: peta attention rollout hampir identik dengan raw attention (r=0,9998) dan **token-invariant** — kurang bisa dipercaya sebagai penjelasan per-kata. Detail lengkap ada di `docs/design_decisions.md` §6.

### 3.1 RISE — metode xAI pembanding (lebih faithful dari rollout)

- **`xai_rise/comparison_table.md`** / **`faithfulness_summary.json`** — tabel 4-metode (rollout, last_layer, random, RISE) berdampingan.
- **`xai_rise/deletion_insertion_test.json`** + **`deletion_insertion_curves.png`** — hasil uji faithfulness RISE (n=12, sama seperti rollout).
- **Temuan kunci**: RISE (metode berbasis perturbasi, bukan atensi) **satu-satunya metode yang signifikan secara statistik** vs random di uji deletion (p=0,027, rollout tetap p=0,269), dan jauh lebih diskriminatif antar-token (r=0,9502 vs 0,9997 rollout).
- **Catatan kehati-hatian**: angka 0,9502 itu sendiri belum diuji signifikansinya secara formal (baru 1 seed, n=12) — yang solid adalah hasil uji deletion-nya, bukan angka token_selectivity secara presisi. Jangan dikutip sebagai angka final tanpa disclaimer ini.

## 4. Verifikasi checkpoint eksternal (bukan kontribusi kita, tapi bukti due-diligence)

- **`external_checkpoint_tokenizer_test.json`** — bukti fragmentasi tokenizer (checkpoint `indoblip-lr-5e-06-epoch-30` pakai tokenizer Inggris, 2,64 token/kata vs 1,20 punya kita).
- **`external_checkpoint_zeroshot_test.json`** — checkpoint eksternal itu gagal total di tugas kita (CIDEr 0,0000, output token rusak berulang).

## 5. Eksperimen backbone alternatif: InternVL3-1B-hf

**Baca dulu `internvl3_experiment.md`** (sudah dikopi ke folder ini) — cerita lengkap dari awal (kenapa dicoba, rintangan teknis, setup) sampai akhir (hasil, narasi efisiensi, rekomendasi pemakaian di paper). Ringkasan filenya:

- **`internvl3_vs_blip_comparison.json`** — perbandingan 3-arah (epoch1/epoch2/BLIP) di 2.362 test set penuh.
- **`internvl3_vs_blip_shared_subset_1182.json`** — perbandingan yang sama tapi di 1182 item identik dengan evaluasi Qwen2.5-VL-3B rekan tim (siap disandingkan 3-arah begitu skor Qwen tersedia).
- **`internvl3_shared_subset_1182_metrics.json`** — skor BLIP saja di subset 1182 (dibuat lebih awal, sebelum InternVL3 ada).

**Temuan kunci**: InternVL3 (0,94B) setelah cuma 2 epoch sudah **setara secara statistik** dengan BLIP (0,224B, 30 epoch) di CIDEr/BLEU-4 (p=0,08-0,10, belum signifikan) — TAPI akurasi klasifikasi jenis bencana masih jauh tertinggal (53,1% vs 72,3%), dan yang paling penting: **butuh waktu wall-clock hampir sama (4j45m) untuk 2 epoch saja, sementara BLIP mencapai 30 epoch penuh sampai konvergen di 4j32m** — InternVL3 ~15,7x lebih lambat per-epoch. Ini jadi narasi efisiensi yang kuat: model kecil yang diadaptasi cermat vs model besar dengan pretraining kuat tapi lambat per-epoch di hardware terbatas.

## 6. Perbandingan draft paper (fiktif) vs hasil real

- **`draft_vs_real_comparison.md`** — perbandingan eksplisit klaim draft paper awal (Table I, Figure 3-5 — semuanya mock-up fiktif, bukan hasil eksperimen) vs angka real dari proyek ini. **Wajib dibaca sebelum menulis bagian hasil di paper**, supaya tidak ada yang salah kutip angka fiktif sebagai hasil asli.

---

## Yang BELUM ada di sini (dan kenapa)

- **Qwen2.5-VL-3B**: masih dievaluasi rekan tim di subset 1182, belum tersedia hasilnya untuk digabung.
- **Notebook lengkap** (EDA sampai xAI, `notebooks/01-06`): tidak dikopi ke sini karena ukurannya besar (gambar ter-embed) — tetap ada di folder `notebooks/` repo utama, sudah tereksekusi dengan output nyata.
- **Prediksi mentah** (`predictions_*.jsonl`, beberapa MB per file): tidak dikopi, ada di `results/` repo utama kalau butuh lihat contoh generasi lengkap.
