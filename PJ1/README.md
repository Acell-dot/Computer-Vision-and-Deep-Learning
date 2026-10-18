
# ANALISIS PJ1 (RET503)
**Pengembangan, Pelatihan, Tuning, Seleksi, dan Pengujian Model YOLO**

---

### Informasi Mahasiswa & Sistem
- **Nama Mahasiswa** : Adfa Viona Marshelia
- **NIM** : 4222101027
- **Mata Kuliah** : Praktik Terpadu Minggu 4–7 (RET503)
- **Model Dasar** : YOLO (yolo26n)
- **Dataset** : `dataset_v1` (31 Kelas Target)
- **Sidik Jari Label** : `EB643308968D`
- **Lingkungan Uji** : ASUS VivoBook Laptop (CPU Mode, 16 Cores)

---

## TAHAP 1–2: Persiapan Dataset & Verifikasi Kualitas Anotasi (Asesmen P3 / CLO-3)

### 1. Tahap 1: Pengumpulan & Standarisasi Dataset Projek
- **Sumber Data**: Dataset awal dihimpun dan distandarisasi dari Roboflow (`dataset_roboflow`) ke struktur baku YOLO (`dataset_yolo`).
- **Struktur Standarisasi**:
  - `images/`: Berisi seluruh citra citra digital objek target.
  - `labels/`: Berisi file label `.txt` dengan penamaan identik terhadap file citra.
  - `classes.txt`: Berisi daftar 31 kelas target (simbol huruf, angka, dan panah warna-warni: *Blue, Green, Red, White, Yellow*).
- **Pembagian Dataset (*Data Splitting*)**:
  Menggunakan script `05_split_dataset_yolo.py` dengan *random seed* tetap (`seed=42`) agar hasil reproduktif, dataset dibagi dengan proporsi **70% Train, 20% Val, 10% Test**:
  - **Data Train** : **972 citra** (digunakan untuk memperbarui bobot model).
  - **Data Val** : **278 citra** (digunakan untuk validasi tiap epoch dan seleksi model terbaik).
  - **Data Test** : **140 citra** (disimpan terisolasi untuk evaluasi final generalisasi).
  - **Total Dataset** : **1.390 citra**.
- **Konfigurasi `data.yaml`**: Jalur dataset absolut, penunjukan direktori split, serta indeks 31 kelas didefinisikan secara otomatis ke `dataset_v1/data.yaml`.

### 2. Tahap 2: Pemeriksaan & Validasi Kualitas Anotasi
Pemeriksaan dilakukan menggunakan `01_cek_anotasi.py` untuk menjamin integritas data sebelum proses pelatihan:
- **Kesesuaian Format**: Memastikan setiap baris anotasi memuat tepat 5 atribut: `<class_id> <x_center> <y_center> <width> <height>`.
- **Normalisasi Koordinat**: Memastikan seluruh nilai koordinat dan ukuran kotak pembatas telah dinormalisasi dalam rentang interval $[0, 1]$.
- **Validasi Kelas**: Memastikan nilai `class_id` berada pada rentang yang valid (`0 <= class_id < 31` atau indeks 0 sampai 30) sesuai `classes.txt`.
- **Pengecekan Anomali**: Mendeteksi jika terdapat ukuran kotak ekstrem ($< 8$ px) atau label kosong/hilang.
- **Verifikasi Visual**: Hasil anotasi digambar ulang (*bounding box overlay*) ke folder `cek_visual/` untuk memastikan secara visual bahwa kotak pembatas tepat membingkai objek target tanpa terpotong.
- **Integritas Dataset**: Hash SHA-256 label tercatat sebagai `EB643308968D` sebagai bukti integritas data projek mahasiswa.

---

## TAHAP 3: Pelatihan Model Eksperimen E1 dan E3 serta Visualisasi Grafik (Asesmen P4 / CLO-4)

Pada tahap ini dijalankan dua variasi eksperimen utama:
1. **E1_pretrained (Baseline)**: Model YOLO26n dengan bobot awal *Pretrained COCO*, dilatih menggunakan resolusi standar `imgsz=416`, `epochs=30`, dan `batch=16`.
2. **E3_tuning (Hyperparameter Tuning)**: Model YOLO26n berbasis bobot Pretrained COCO dengan **SATU perubahan hyperparameter**, yaitu menaikkan resolusi citra input menjadi `imgsz=640` (`epochs=30`, `batch=16`).

### 1. Tabel Perbandingan Hasil Tahap 3 (E1 vs E3)

| Parameter / Metrik | Eksperimen E1 (Baseline) | Eksperimen E3 (Tuning) | Selisih / Dampak Perubahan |
| :--- | :---: | :---: | :--- |
| **Konfigurasi Diubah** | - (Baseline) | `imgsz=640` | Resolusi naik dari 416 ke 640 px |
| **Transfer Learning** | Ya (Pretrained COCO) | Ya (Pretrained COCO) | Sama-sama memanfaatkan transfer bobot |
| **mAP@0.5** | **0.995** (99.5%) | **0.995** (99.5%) | Akurasi deteksi kasar setara |
| **mAP@0.5:0.95** | **0.889** (88.9%) | **0.898** (89.8%) | **+0.9%** (Presisi lokalisasi kotak meningkat) |
| **Precision (P)** | 0.984 (98.4%) | **0.987** (98.7%) | **+0.3%** (Deteksi positif palsu berkurang) |
| **Recall (R)** | **0.989** (98.9%) | **0.989** (98.9%) | Konsisten mendeteksi seluruh target |
| **Waktu Inferensi** | **30.8 ms** / citra | 80.4 ms / citra | Naik 2.6x lebih lambat karena piksel lebih padat |
| **Waktu Pelatihan** | **115.3 menit** (~1.9 jam) | 283.8 menit (~4.7 jam) | Naik 2.4x lebih lama |

### 2. Analisis Grafik Pelatihan (`results.png` pada E1 dan E3)
Grafik metrik lengkap tersimpan pada:
- **E1**: `hasil_praktik/E1_pretrained/results.png`
- **E3**: `hasil_praktik/E3_tuning/results.png`

**Pembahasan Karakteristik Kurva:**
1. **Konvergensi Loss (`train/box_loss`, `train/cls_loss`, `val/box_loss`, `val/cls_loss`)**:
   - Baik pada E1 maupun E3, *classification loss* (`cls_loss`) menurun sangat tajam pada 5 epoch pertama (dari $\approx 39.4$ turun ke $< 5.0$), menandakan model sangat cepat mengenali kategori 31 simbol berkat bobot pretrained.
   - Kurva *loss* data validasi berjalan beriringan dan stabil mengikuti kurva training tanpa lonjakan divergence, membuktikan proses pembelajaran berjalan lancar.
2. **Pertumbuhan Metrik Akurasi (`metrics/mAP50(B)` dan `metrics/mAP50-95(B)`)**:
   - Pada E1 (`imgsz=416`), mAP50 mencapai titik jenuh mendekati $1.0$ sekitar epoch ke-15.
   - Pada E3 (`imgsz=640`), resolusi yang lebih tinggi memberikan keuntungan pada kurva mAP50-95 yang terus menanjak stabil hingga menyentuh **0.898** di akhir epoch ke-30. Hal ini disebabkan kerapatan piksel pada ukuran 640 memungkinkan gradien garis dan sudut simbol kecil terhitung lebih presisi saat perhitungan IoU bertingkat (0.5 hingga 0.95).

---

## TAHAP 4–6: Seleksi Model, Uji Generalisasi Data TEST, dan Analisis Akhir (Asesmen Pj1 / CLO-4)

### 1. Tahap 4: Pemilihan Model Terbaik (Berdasarkan Data Val)
Untuk menguji signifikansi metode, dilakukan pembandingan tiga eksperimen secara komprehensif:

```
  Eksperimen     Perubahan        mAP50  mAP50-95      P      R     ms   menit
  E1_pretrained  -                0.995     0.889  0.984  0.989   30.8   115.3
  E2_scratch     -                0.637     0.536  0.614  0.580   30.1   110.4
  E3_tuning      imgsz=640        0.995     0.898  0.987  0.989   80.4   283.8
```

- **Kriteria Pemilihan**: Model terbaik dipilih berdasarkan nilai **mAP50-95 tertinggi pada data Validasi (VAL)**, karena mAP50-95 mengukur ketepatan deteksi sekaligus presisi posisi *bounding box* di berbagai ambang batas IoU (50% hingga 95%).
- **Model Terpilih**: **`E3_tuning`** dengan skor validasi **mAP50-95 = 0.898**.

---

### 2. Tahap 5: Pengujian Model Terpilih pada Data TEST (Unseen Data)
Sesuai metodologi baku Machine Learning, model terbaik (`E3_tuning/weights/best.pt`) diuji **satu kali saja** menggunakan kumpulan data uji independen (`test` set berisi 140 citra) yang sama sekali tidak pernah disentuh selama pelatihan maupun tuning:

- **mAP@0.5 (Data TEST)** : **0.995** (99.5%)
- **mAP@0.5:0.95 (Data TEST)** : **0.882** (88.2%)

**Evaluasi Uji TEST:**
Performa pada data Test (0.882) menunjukkan konsistensi yang sangat tinggi terhadap performa data Val (0.898), hanya berselisih sangat tipis sebesar $0.016$ ($1.6\%$). Hal ini membuktikan bahwa:
- Model memiliki **daya generalisasi yang sangat kuat** terhadap citra baru.
- Model **tidak mengalami overfitting** (*hafalan*) terhadap data latih.

---

### 3. Tahap 6: Analisis Mendalam & Diskusi Komprehensif

#### A. Pengaruh Transfer Learning (E1 Pretrained vs E2 Scratch)
- Model E1 (Pretrained) menghasilkan mAP50-95 sebesar **0.889**, sedangkan E2 (Scratch) hanya memperoleh **0.536** (terdapat penurunan performa drastis sebesar **-35.3%**).
- **Pembahasan Ilmiah**: Pada E2, bobot jaringan diinisialisasi secara acak. Dalam 30 epoch dan dataset berjumlah $\approx 1.000$ citra, konvergensi fitur representasi objek belum tercapai secara maksimal. Sebaliknya, E1 memanfaatkan bobot *transfer learning* dari ribuan kelas COCO sehingga model sudah mewarisi kemampuan ekstraksi fitur tingkat rendah (*low-level features* seperti tepi, sudut, kontras, dan kurva) sejak epoch pertama.

#### B. Pengaruh Hyperparameter Tuning (E1 Baseline vs E3 Tuning)
- Menaikkan ukuran citra dari 416 menjadi 640 piksel berhasil mendongkrak mAP50-95 dari **0.889 ke 0.898** dan Precision dari **0.984 ke 0.987**.
- **Analisis Trade-off Komputasi**:
  - **Keuntungan**: Objek-objek simbol dengan bentuk rumit (seperti *Bullseye*, *Arrow*, dan huruf kecil) memiliki resolusi spasial lebih kaya sehingga lokalisasi kotak lebih rapat.
  - **Konsekuensi**: Kebutuhan memori dan komputasi floating-point bertambah drastis. Waktu komputasi inferensi melambat dari **30.8 ms** menjadi **80.4 ms** per gambar, dan waktu latih membengkak dari **115.3 menit** menjadi **283.8 menit**.

#### C. Rekomendasi Implementasi Praktis
- **Untuk Kebutuhan Edge / Real-Time (misal: Robotika / Kamera Bergerak)**: Model **`E1_pretrained`** lebih direkomendasikan karena memberikan kecepatan inferensi hingga $\approx 32.5$ FPS (30.8 ms) dengan akurasi yang sudah sangat memuaskan (mAP 88.9%).
- **Untuk Kebutuhan Akurasi Maksimal (misal: Sistem Inspeksi Kualitas / Server-side)**: Model **`E3_tuning`** adalah pilihan terbaik karena memberikan tingkat presisi dan lokalisasi batas objek tertinggi (mAP 89.8%).

---

## 4. Berkas Rekapitulasi & Artefak Pengumpulan
Seluruh berkas luaran tersimpan secara otomatis dan valid di direktori kerja:
1. Rekapitulasi format teks : `hasil_praktik_4222101027.txt`
2. Rekapitulasi format tabel : `hasil_praktik_4222101027.csv`
3. Grafik evaluasi E1 : `hasil_praktik/E1_pretrained/results.png`
4. Grafik evaluasi E3 : `hasil_praktik/E3_tuning/results.png`
5. Bobot model terbaik : `hasil_praktik/E3_tuning/weights/best.pt`
6. Dokumen laporan analisis : `README.md`
