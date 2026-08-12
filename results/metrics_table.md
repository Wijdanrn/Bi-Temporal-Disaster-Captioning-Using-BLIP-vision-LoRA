# Fase 4-5 -- Final Real Evaluation Results

**Final model checkpoint:** `checkpoints/full_run_v3_vision/best` (text LoRA + vision qkv LoRA + trained embeddings, val_loss 1.4312)

Test set: 2362 / 2,362 usable records scored

## Aggregate metrics

| Metric | Zero-shot BLIP baseline | Text-only LoRA (ablation) | **Text+Vision LoRA (final)** |
|---|---|---|---|
| CIDEr | 0.0001 | 0.0980 | **0.1202** |
| BLEU-4 | 0.0000 | 0.2044 | **0.2095** |
| ROUGE-L | 0.0185 | 0.4103 | **0.4156** |
| BERTScore F1 (SPICE substitute) | 0.3470 | 0.7943 | **0.7971** |

## Diagnostic checks (Fase 5 -- disaster-type accuracy & no-damage bias)

| Diagnostic | Text-only LoRA | **Text+Vision LoRA (final)** | Real reference rate |
|---|---|---|---|
| BENCANA (disaster type) exact-match accuracy | 58.8% | **72.3%** | -- |
| BANGUNAN "no damage" boilerplate rate | 83.0% | **78.5%** | 34.2% |
| Distinct generated captions | 87.8% | 86.5% | -- |

SPICE itself was NOT computed (no Indonesian WordNet coverage) -- see docs/design_decisions.md SS4.1 for the substitution decision.

## Statistical significance (paired bootstrap, 10,000 resamples, per-example CIDEr)

**Final model vs. zero-shot baseline:**
- Observed mean CIDEr difference: **0.1201**
- 95% CI: [0.1078, 0.1329]
- Approx. two-sided p-value: 0

**Final model (+ vision LoRA) vs. text-only LoRA ablation:**
- Observed mean CIDEr difference: **0.0221**
- 95% CI: [0.0099, 0.0345]
- Approx. two-sided p-value: 0.0004
- Paired N: 2362

## Qualitative samples (final model)

### index 456
**Reference (canonicalized):**
```
BENCANA: tornado
BANGUNAN: tidak ada perubahan struktural yang terlihat pada bangunan atau fasilitas antara gambar sebelum dan sesudah bencana.
JALAN: jaringan jalan tetap utuh, tanpa pemblokiran, kerusakan, atau gangguan yang terlihat.
VEGETASI: penutup vegetasi tampak konsisten, tanpa hilangnya jelas, berakar, atau perubahan dalam refleksi spektral yang menunjukkan kerusakan pada penutup tanaman alami.
BADAN_AIR: tidak ada benda air yang dapat diidentifikasi atau perubahan dalam kehadiran, ukuran, atau bentuk air terlihat dalam gambar.
PERTANIAN: tidak ada lahan pertanian yang terlihat, pola
```
**Generated (text+vision LoRA):**
```
BENCANA: badai
BANGUNAN: tidak ada kerusakan struktural yang terlihat pada bangunan; atap dan tata letak tetap konsisten antara gambar sebelum dan sesudah bencana.

JALAN: tidak ada gangguan, pemblokiran, atau banjir yang terlihat di jaringan jalan.
VEGETASI: vegetasi alami, termasuk penutup pohon dan daerah tumbuhan yang tidak dikelola, tidak menunjukkan perubahan yang terlihat dalam refleksi spektral atau cakupan antara gambar sebelum dan sesudah bencana.
BADAN_AIR: tidak ada perubahan yang terlihat dalam ukuran, bentuk, atau kehadiran benda air.
PERTANIAN: tidak ada lahan pertanian yang dik
```

### index 102
**Reference (canonicalized):**
```
BENCANA: ledakan
BANGUNAN: struktur dan bangunan kota tidak menunjukkan tanda-tanda kerusakan struktural yang terlihat, dengan pengaturan spasial yang konsisten dan pola refleksi antara gambar sebelum dan sesudah bencana.
JALAN: jaringan jalan tetap utuh tanpa penghalang yang terlihat, deformasi, atau puing-puing yang diamati dalam gambar pasca bencana jika dibandingkan dengan kondisi sebelum bencana.
VEGETASI: tutupan vegetasi, terlihat di bagian bawah kanan gambar, mempertahankan tanda tangan spektrum dan pola spasial yang konsisten, menunjukkan tidak ada kerusakan atau kerugian yang dapat d
```
**Generated (text+vision LoRA):**
```
BENCANA: ledakan
BANGUNAN: tidak ada kerusakan struktural yang terlihat atau perubahan pada bangunan. semua bangunan tampak utuh dan tidak berubah antara gambar sebelum dan sesudah bencana.
JALAN: tidak ada gangguan, pemblokiran, atau perubahan yang terlihat di jaringan jalan.
VEGETASI: tidak ada perubahan yang terlihat pada lapisan vegetasi alami; refleksi spektral dan penutup tanaman tetap konsisten di kedua gambar.
BADAN_AIR: tidak ada perubahan yang terlihat dalam ukuran, bentuk, atau kehadiran benda air yang terdeteksi dalam gambar.
PERTANIAN: tidak ada lahan pertanian yang dapat diidenti
```

### index 1126
**Reference (canonicalized):**
```
BENCANA: gempa bumi
BANGUNAN: tidak ada bangunan yang terlihat di kedua gambar.
JALAN: struktur jetty yang terlihat di kedua gambar tetap utuh dan tidak menunjukkan tanda-tanda kerusakan fisik atau penghalang.
VEGETASI: tidak ada tumbuh-tumbuhan alami yang terlihat di gambar-gambar ini.
BADAN_AIR: badan air mempertahankan batas yang konsisten dan pola refleksi spektral antara gambar sebelum dan sesudah bencana, yang menunjukkan tidak ada perubahan yang signifikan.
PERTANIAN: tidak ada ladang pertanian atau lahan yang dikelola yang hadir dalam gambar.
KESIMPULAN: gempa bumi tampaknya memiliki d
```
**Generated (text+vision LoRA):**
```
BENCANA: banjir
BANGUNAN: tidak ada bangunan yang terlihat dalam gambar sebelum atau sesudah bencana.
JALAN: tidak ada jaringan jalan yang terlihat dalam gambar sebelum atau sesudah bencana.
VEGETASI: gambar sebelum bencana menunjukkan vegetasi alami yang padat dengan refleksi spektral yang konsisten dengan vegetasi yang sehat.
BADAN_AIR: tidak ada benda air yang terlihat dalam gambar sebelum atau sesudah bencana.
PERTANIAN: tidak ada lahan pertanian atau lahan yang dikelola yang terlihat dalam gambar.
KESIMPULAN: banjir menyebabkan kerusakan struktural kecil pada bangunan dan hilangnya vegeta
```

### index 1003
**Reference (canonicalized):**
```
BENCANA: tornado
BANGUNAN: semua bangunan yang terlihat dalam gambar sebelum dan sesudah bencana tampak secara struktural utuh, tanpa perpindahan yang jelas, puing-puing, atau perubahan struktural.
JALAN: jaringan jalan tetap tidak berubah, tanpa pemblokiran yang terlihat, gangguan permukaan, atau perubahan antara gambar sebelum dan sesudah bencana.
VEGETASI: vegetasi di daerah ini menunjukkan pola refleksi spektral yang konsisten antara gambar sebelum dan sesudah bencana. tidak ada bukti hilangnya vegetasi, pembakaran, atau perpindahan yang diamati.
BADAN_AIR: benda-benda air yang terlihat da
```
**Generated (text+vision LoRA):**
```
BENCANA: tornado
BANGUNAN: tidak ada kerusakan struktural yang terlihat pada bangunan; semua struktur tampak utuh dan tidak berubah antara gambar sebelum dan sesudah bencana.
JALAN: jaringan jalan tetap utuh tanpa pemblokiran, gangguan, atau perubahan yang terlihat.
VEGETASI: vegetasi di daerah yang tidak dikelola, termasuk penutup tanaman yang tidak dikelola, tidak menunjukkan perubahan yang terlihat dalam refleksi spektral atau cakupan antara gambar sebelum dan sesudah bencana.
BADAN_AIR: tidak ada perubahan yang terlihat dalam ukuran, bentuk, atau kehadiran benda air yang terdeteksi dalam g
```

### index 914
**Reference (canonicalized):**
```
BENCANA: banjir
BANGUNAN: tidak ada kerusakan yang terlihat pada bangunan dalam gambar setelah bencana. struktur tampak utuh tanpa banjir, runtuh, atau pergeseran struktur yang terlihat.
JALAN: jalan tetap terlihat dan tidak terganggu dalam gambar pasca bencana. tidak ada indikasi banjir, penghalang, atau deformasi jaringan transportasi.
VEGETASI: vegetasi alami, termasuk penutup pohon dan daerah hijau lainnya yang tidak dikelola, tidak menunjukkan hilangnya yang nyata atau perubahan spektrum yang terkait dengan banjir atau degradasi.
BADAN_AIR: badan air dalam area yang diamati mempertahankan
```
**Generated (text+vision LoRA):**
```
BENCANA: banjir
BANGUNAN: tidak ada kerusakan struktural yang terlihat pada bangunan; semua struktur tetap utuh dan tidak berubah antara gambar sebelum dan sesudah bencana. JALAN: tidak ada gangguan, pemblokiran, atau banjir yang terlihat di jaringan jalan.
VEGETASI: vegetasi alami, termasuk penutup tanaman yang tidak dikelola, tidak menunjukkan perubahan yang terlihat dalam refleksi spektral atau cakupan antara gambar sebelum dan sesudah bencana.
BADAN_AIR: tidak ada perubahan yang terlihat dalam ukuran, bentuk, atau kehadiran benda air yang terdeteksi dalam gambar.
PERTANIAN: tidak ada lahan
```

### index 571
**Reference (canonicalized):**
```
BENCANA: banjir
BANGUNAN: gambar-gambar menunjukkan degradasi yang terlihat dari infrastruktur yang dibangun, terutama di bagian atas pusat dan bagian bawah.
JALAN: beberapa segmen jalan, terutama di daerah pusat atas, menunjukkan liputan sebagian dengan sedimen dan air berdiri. beberapa jalur tetap utuh tetapi tidak terlihat, yang menunjukkan hilangnya fungsi sementara.
VEGETASI: tidak ada perubahan signifikan yang diamati pada lapisan vegetatif, dengan vegetasi alami mempertahankan karakteristik spasial dan spektral yang konsisten.
BADAN_AIR: tidak ada perubahan pada benda air yang terdeteks
```
**Generated (text+vision LoRA):**
```
BENCANA: banjir
BANGUNAN: tidak ada kerusakan struktural yang terlihat atau perubahan pada bangunan di seluruh area yang dinilai.
JALAN: jaringan jalan tetap utuh tanpa pemblokiran, banjir, atau gangguan yang terlihat.
VEGETASI: vegetasi di daerah alam, termasuk daerah hutan dan padang rumput, tidak menunjukkan perubahan yang terlihat dalam refleksi spektral atau cakupan antara gambar sebelum dan sesudah bencana.
BADAN_AIR: tidak ada perubahan yang terlihat dalam ukuran, bentuk, atau kehadiran benda air yang terdeteksi dalam gambar.
PERTANIAN: tidak ada daerah pertanian yang dapat diidentifika
```

### index 419
**Reference (canonicalized):**
```
BENCANA: tornado
BANGUNAN: tidak ada kerusakan struktural yang terlihat di lingkungan yang dibangun. bangunan dalam gambar tetap utuh tanpa gangguan yang jelas pada atap atau konfigurasi bangunan secara keseluruhan.
JALAN: jaringan jalan tidak menunjukkan indikasi pemblokiran, gangguan permukaan, atau puing-puing, dan kontinuitas spasial mereka tetap tidak berubah.
VEGETASI: tumbuhan tampak konsisten antara gambar sebelum dan sesudah bencana. penandatanganan spektral menunjukkan tidak ada kerugian, pembakaran, atau gangguan pada penutup tanaman alami di daerah itu.
BADAN_AIR: tidak ada perubah
```
**Generated (text+vision LoRA):**
```
BENCANA: badai
BANGUNAN: tidak ada kerusakan struktural yang terlihat pada bangunan; atap dan tata letak tetap konsisten antara gambar sebelum dan sesudah bencana.

JALAN: tidak ada gangguan, pemblokiran, atau kerusakan yang terlihat di jaringan jalan. jalan tetap utuh dan tidak terhalang dalam gambar sebelum dan sesudah bencana.
VEGETASI: tidak ada perubahan yang terlihat pada lapisan vegetasi alami; tanda tangan spektral dan distribusi spasial tetap konsisten di kedua gambar.
BADAN_AIR: tidak ada perubahan yang terlihat dalam ukuran, bentuk, atau kehadiran benda air yang terdeteksi dalam gam
```

### index 2233
**Reference (canonicalized):**
```
BENCANA: gempa bumi
BANGUNAN: tidak ada bangunan yang terlihat di area yang dianalisis.
JALAN: tidak ada jalan yang terlihat dalam gambar yang diamati.
VEGETASI: perubahan spektral kecil terlihat di daerah vegetasi, terutama di bagian tengah dan bawah gambar. perubahan ini dapat menunjukkan gangguan permukaan lokal atau paparan tanah, mungkin karena gempa bumi. namun, luas dan kontinuitas vegetasi alami tetap utuh tanpa kerugian yang signifikan.
BADAN_AIR: tidak ada benda air yang terlihat baik dalam gambar sebelum bencana atau pasca bencana.
PERTANIAN: bidang pertanian di daerah yang dianalis
```
**Generated (text+vision LoRA):**
```
BENCANA: badai
BANGUNAN: tidak ada kerusakan struktural yang terlihat pada bangunan; semua struktur tetap utuh dan tidak berubah antara gambar sebelum dan sesudah bencana. JALAN: tidak ada gangguan atau pemblokiran yang terlihat pada jaringan jalan; jalur tetap utuh dan tidak terhalang.
VEGETASI: tidak ada perubahan yang terlihat pada lapisan vegetasi alami; refleksi spektral dan penutup tanaman alami tetap konsisten di kedua gambar.
BADAN_AIR: tidak ada perubahan yang terlihat dalam ukuran, bentuk, atau kehadiran benda air yang terdeteksi antara gambar sebelum dan sesudah bencana.
PERTANIAN: 
```
