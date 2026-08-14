# Apakah Kontribusi Kami Kuat Tanpa xAI?

**Jawaban singkat: ya, kontribusi inti tetap kuat tanpa bagian xAI.** xAI (rollout + RISE) itu *bonus* yang menambah kedalaman, bukan fondasi yang menopang validitas hasil utama. Kalau terpaksa dipotong karena keterbatasan ruang/waktu, paper masih punya cerita lengkap dan solid tanpa itu.

Dokumen ini menjabarkan kenapa, dengan bukti konkret dari yang sudah kita kerjakan — bukan klaim kosong.

---

## Kelebihan yang berdiri sendiri (tidak butuh xAI untuk valid)

### 1. Verifikasi dataset yang jujur dan sistematis
Kami tidak menerima klaim dataset (bahkan draft paper kami sendiri) begitu saja — diaudit langsung dari file mentah, ditemukan mismatch nyata (bi-temporal pairs 12.861 vs klaim 26.988; split captioning 7.766/2.363 vs klaim 17.190/5.024), dilaporkan apa adanya lewat dua metode pengukuran independen yang sepakat. Ini bukan kesalahan kami — dan kejujuran melaporkannya adalah kekuatan metodologis, bukan kelemahan.

### 2. Solusi teknis konkret untuk masalah nyata: fragmentasi tokenizer
Bukan klaim kualitatif ("tokenizer Inggris kurang cocok") — kami **mengukur** langsung (2,64 → 1,20 token/kata, perbaikan 2,19×) dan membangun solusi lengkap (vocab swap + warm-start embedding + resize), diverifikasi round-trip. Ini reusable sebagai metodologi untuk adaptasi VLM captioning ke bahasa low-resource manapun, tidak terikat ke satu dataset.

### 3. Diagnosis akar masalah yang disiplin, bukan tembak-menembak
Saat model menunjukkan bias, kami tidak langsung lompat ke solusi mahal. Kontrol murah dulu (beam search vs greedy) untuk menyingkirkan hipotesis decoding **sebelum** berinvestasi pada eksperimen vision-LoRA yang lebih mahal. Ini pola kerja ilmiah yang jarang terlihat eksplisit di paper kompetisi — biasanya orang langsung klaim "kami tambahkan X, hasilnya membaik" tanpa membuktikan X memang penyebabnya.

### 4. Hasil real dengan uji statistik, metodologi ditetapkan sebelum eksperimen
CIDEr/BLEU-4/ROUGE-L/BERTScore dengan *paired bootstrap significance testing* di semua perbandingan penting — bukan cuma angka mentah ditaruh di tabel. Metodologi evaluasi (termasuk substitusi SPICE dengan BERTScore Indonesia, karena SPICE butuh WordNet yang tidak ada untuk Bahasa Indonesia) ditetapkan dan didokumentasikan **sebelum** hasil dilihat.

### 5. Perbaikan bias yang terukur DAN kejujuran soal keterbatasannya
Akurasi klasifikasi jenis bencana naik signifikan (58,8%→72,3%) setelah LoRA vision ditambahkan. Tapi kami tidak berhenti di situ dengan klaim "masalah selesai" — rate "tidak ada kerusakan" (78,5%) masih jauh dari rate referensi asli (34,2%), dan ini dilaporkan eksplisit sebagai limitasi. Kejujuran soal keberhasilan *parsial* (bukan klaim sempurna) justru memperkuat kredibilitas ilmiah.

### 6. Studi banding backbone dengan temuan yang genuinely menarik
Bukan sekadar "kami coba model lain, hasilnya mirip" — kami menemukan dan mengukur **trade-off efisiensi yang tajam dan spesifik**: dalam anggaran waktu wall-clock yang hampir sama (~4,5 jam), BLIP (224M) mencapai konvergensi penuh 30 epoch, sementara InternVL3 (938M, 4,19× lebih besar) baru 2 epoch dan belum konvergen — meski performanya per-epoch memang lebih cepat "mengejar". Ini temuan yang berdiri sendiri, relevan untuk siapa pun yang meneliti dengan sumber daya terbatas (bukan cuma konteks proyek kami).

### 7. Semuanya dikerjakan di satu GPU laptop 8GB
Seluruh pipeline — training, evaluasi, ablasi, studi banding backbone — dijalankan di satu RTX 5050 laptop 8GB, bukan cluster atau cloud multi-GPU. Ini poin aksesibilitas nyata: menunjukkan riset VLM adaptasi bahasa yang serius tetap bisa dilakukan dengan sumber daya mahasiswa/kompetisi, bukan cuma lab besar.

### 8. Infrastruktur reproducibility yang lengkap
6 notebook (EDA sampai xAI) yang benar-benar dieksekusi dengan output nyata, log training lengkap, checkpoint tersimpan, dokumentasi alasan tiap keputusan desain (`design_decisions.md`, >1000 baris). Ini kekuatan yang jarang ditonjolkan tapi sangat dihargai reviewer — bukti bahwa hasil bisa diverifikasi ulang, bukan klaim yang tidak bisa dicek.

---

## Yang hilang kalau xAI benar-benar dipotong (untuk keseimbangan, jangan diabaikan)

- Tidak ada jawaban untuk pertanyaan "bagaimana kita tahu model benar-benar melihat area yang relevan?" — bagian eksplainabilitas biasanya jadi cara menjawab kekhawatiran "black box" ini.
- Kehilangan satu sudut kontribusi tambahan: perbandingan metode xAI (rollout vs RISE) itu sendiri temuan yang solid (RISE terbukti signifikan lebih faithful, p=0,027) — related work yang relevan (RS-LLaVA, GeoChat, dll.) jarang membahas eksplainabilitas sama sekali, jadi ini pembeda yang cukup unik kalau dipertahankan.
- Cakupan paper jadi sedikit lebih sempit (5 bagian solid vs 6), tapi **tidak ada satu pun klaim di bagian lain yang bergantung pada temuan xAI untuk tetap valid** — beda dengan misalnya §5.3 (diagnosis bias) yang justru bergantung pada §5.2 (hasil utama).

---

## Rekomendasi

Kalau ruang/waktu terbatas dan harus memilih prioritas: **jangan korbankan §5 (hasil utama + diagnosis bias) atau §6 (studi banding InternVL3)** — dua ini paling kuat dan saling bergantung. xAI (§7) aman dipangkas jadi ringkasan singkat (2-3 kalimat + 1 figure saja, mis. `fig06` atau `fig08` dari `paper_figures/`) atau dipindah ke lampiran kalau memang kepepet, tanpa merusak argumen inti paper.
