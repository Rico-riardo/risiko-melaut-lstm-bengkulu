# Peringatan Dini Risiko Melaut Perahu Nelayan di Perairan Bengkulu (LSTM + Kriteria BMKG)

Kode pendukung artikel:

> Rico Riardo dan Harry Witriyono. *Evaluasi Peringatan Dini Risiko Melaut Perahu Nelayan Berbasis Ramalan LSTM dan Kriteria Gabungan BMKG di Perairan Bengkulu.* Program Studi Teknik Informatika, Fakultas Teknik, Universitas Muhammadiyah Bengkulu (draf, 2026).

Repositori ini berisi skrip untuk (1) mengunduh data ERA5 dari Copernicus Climate Data Store, (2) membentuk data per jam dan harian di tiga titik perairan Bengkulu beserta label risiko menurut kriteria BMKG untuk perahu nelayan, (3) melatih LSTM dan membandingkannya dengan metode persistensi, dan (4) menguji perbaikan berupa kalibrasi ambang, keluaran peluang, dan masukan spasial.

## Data

| Keterangan | Nilai |
|---|---|
| Sumber | ERA5 hourly data on single levels (Copernicus Climate Data Store) |
| Variabel | `u10`, `v10` (angin 10 m, grid 0,25°) dan `swh` (tinggi gelombang signifikan, grid 0,5°) |
| Wilayah | 2° LS – 5,5° LS; 100,5° BT – 103,5° BT |
| Periode | 1 Januari 2016 – 31 Desember 2025, per jam (87.672 jam) |
| Titik penelitian | utara (2,5° LS; 100,5° BT), kota (4,0° LS; 102,0° BT), kaur (5,0° LS; 103,0° BT) |
| Pembagian | latih 2016–2023, validasi 2024, uji 2025 |

Data mentah (±104 MB, 40 file NetCDF) **tidak** disimpan di repositori. Jalankan `unduh_era5.py` untuk mengunduhnya kembali.

Kriteria risiko (BMKG, perahu nelayan): suatu jam berisiko bila kecepatan angin > 15 knot **dan** tinggi gelombang signifikan > 1,25 m; suatu hari (WIB) berisiko bila ada minimal satu jam berisiko.

## Struktur folder

```
01_data_mentah/   file ERA5 hasil unduhan (ERA5_<tahun>/era5_{angin,gelombang}_<tahun>_{jan-jun,jul-des}.nc)
02_data_olahan/   perjam_3lokasi.csv dan harian_3lokasi.csv
03_skrip/         skrip Python
04_hasil/         tabel metrik, ramalan, grafik, dan model
```

## Cara menjalankan

1. Buat environment dan pasang paket:
   ```
   conda create -n lstm python=3.11
   conda activate lstm
   pip install -r requirements.txt
   ```
2. Buat akun Copernicus Climate Data Store, setujui lisensi ERA5, lalu simpan Personal Access Token di file `.cdsapirc` di folder pengguna (lihat petunjuk di awal `unduh_era5.py`). File ini bersifat rahasia dan tidak boleh diunggah.
3. Jalankan berurutan dari folder utama repositori:
   ```
   python 03_skrip/unduh_era5.py
   python 03_skrip/praproses_3lokasi.py
   python 03_skrip/latih_lstm.py
   python 03_skrip/uji_perbaikan_lstm.py . 04_hasil/uji_perbaikan
   ```

| Skrip | Fungsi | Keluaran utama |
|---|---|---|
| `unduh_era5.py` | mengunduh ERA5 per paruh tahun; file yang sudah ada dilewati | `01_data_mentah/` |
| `praproses_3lokasi.py` | ekstraksi titik, konversi m/s ke knot, agregasi harian WIB, label risiko, pemeriksaan mutu | `02_data_olahan/*.csv`, `04_hasil/ringkasan_praproses.csv` |
| `latih_lstm.py` | LSTM per lokasi (jendela 14 hari, ramalan H+1 sampai H+3) dibandingkan dengan persistensi | `04_hasil/lstm/` (RMSE, recall, precision, FAR, grafik) |
| `uji_perbaikan_lstm.py` | LSTM lokal dan spasial (PCA), keluaran nilai dan peluang, kalibrasi ambang pada data validasi, tiga benih acak | `04_hasil/uji_perbaikan/` (CSI) |

Hasil di artikel diperoleh dengan Python 3.11 dan TensorFlow 2.21 pada CPU (Windows). Karena perbedaan perhitungan numerik antarperangkat, nilai yang dihasilkan di komputer lain dapat berbeda sedikit (sekitar ±0,01 pada CSI).

## Sitasi data

Hersbach, H. dkk. (2023). *ERA5 hourly data on single levels from 1940 to present.* Copernicus Climate Change Service (C3S) Climate Data Store. doi:10.24381/cds.adbb2d47

## Lisensi

Kode dirilis dengan lisensi MIT (lihat `LICENSE`). Data ERA5 tunduk pada lisensi Copernicus.
