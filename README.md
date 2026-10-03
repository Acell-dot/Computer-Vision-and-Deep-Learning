# Computer-Vision-and-Deep-Learning

#Klasifikasi Gambar menggunakan ResNet-50 (Transfer Learning - Mode Partial)

Repository ini berisi dokumentasi, hasil eksperimen, dan kode untuk praktikum klasifikasi gambar menggunakan arsitektur ResNet-50 dengan pendekatan Transfer Learning dalam mode Partial.

#Struktur dataset
Dataset yang digunakan dalam eksperimen ini adalah dataset yang terdiri dari dua kelas target:
- **Huruf_angka** : Gambar yang berisi karakter huruf atau angka.
- **Panah** : Gambar yang berisi simbol penunjuk arah (panah).

#Statistik DIstribusi data:
- **Total Data Pelatihan (Train):** 287 gambar (80%)
- **Total Data Validasi (Val):** 72 gambar (20%)

#Hasil Pelatihan Model
Model dilatih selama **10 Epoch** 

### Ringkasan Performa:
- **Akurasi Validasi Terbaik:** **97.2%**
- **Tercapai pada Epoch:** 6
- **Waktu Pelatihan:** ~29 detik (Menggunakan T4 GPU di Google Colab)

---

## ⏱️ Hasil Pengujian Latensi Inferensi
Pengukuran latensi dilakukan untuk memproses satu gambar masukan ($224 \times 224$ piksel) guna menguji kelayakan integrasi *real-time* pada robot **Smorphi**:
| Perangkat (Device) | Waktu Inferensi per Gambar | Perkiraan FPS | Status Target (Anggaran 35 ms) |
| :--- | :--- | :--- | :--- |
| **GPU (Tesla T4)** | **6.6 ms - 7.3 ms** | ~137.9 - 152.1 FPS | **MUAT** (Sangat Layak) |
| **CPU (Colab)** | **125.6 ms - 129.3 ms** | ~7.7 - 8.0 FPS | **TIDAK MUAT** (Terlalu Lambat) |

---

## 📦 Paket Hasil Eksperimen
Semua metrik dan visualisasi dikompresi dalam berkas **`hasil_resnet50_partial_kustom.zip`** yang berisi:
1. `hasil_partial.png` - Grafik tren akurasi per epoch & Confusion Matrix validasi.
2. `hasil_partial.json` - Metadata spesifikasi model, waktu latih, dan latensi detail.
3. `history_partial.csv` - Log nilai loss dan akurasi tiap epoch.
