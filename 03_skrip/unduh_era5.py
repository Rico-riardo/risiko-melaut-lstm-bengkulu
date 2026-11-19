"""Unduh data ERA5 (angin u10, v10 dan tinggi gelombang swh) perairan Bengkulu.

Penelitian: Evaluasi Peringatan Dini Risiko Melaut Perahu Nelayan Berbasis
Ramalan LSTM dan Kriteria Gabungan BMKG di Perairan Bengkulu (Rico Riardo).

Cara pakai (dari folder D:\\ARTIKEL_ILMIAH\\Data_Artikel):
    python 03_skrip\\unduh_era5.py

Syarat:
    1. pip install cdsapi
    2. File .cdsapirc di folder pengguna (C:\\Users\\<nama>\\.cdsapirc) berisi:
           url: https://cds.climate.copernicus.eu/api
           key: <Personal Access Token dari halaman profil Copernicus>
    3. Lisensi ERA5 sudah disetujui di situs Copernicus (sudah dilakukan).

Skrip melewati file yang sudah ada, jadi aman dijalankan ulang kalau terputus.
"""
import os
import sys
import time
import shutil
import zipfile
import tempfile

import cdsapi

TAHUN = list(range(2016, 2026))           # 2016-2025; file yang sudah ada dilewati
PARUH = {"jan-jun": range(1, 7), "jul-des": range(7, 13)}
AREA = [-2, 100.5, -5.5, 103.5]          # North, West, South, East (sama dengan data 2024)
FOLDER_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "01_data_mentah")


def target(tahun, paruh, jenis):
    return os.path.join(FOLDER_DATA, f"ERA5_{tahun}", f"era5_{jenis}_{tahun}_{paruh}.nc")


def sudah_ada(tahun, paruh):
    return all(os.path.exists(target(tahun, paruh, j)) and os.path.getsize(target(tahun, paruh, j)) > 100_000
               for j in ("angin", "gelombang"))


def susun_ulang(file_unduhan, tahun, paruh):
    """CDS mengirim zip berisi dua file (oper = angin, wave = gelombang). Ekstrak dan beri nama rapi."""
    os.makedirs(os.path.dirname(target(tahun, paruh, "angin")), exist_ok=True)
    if not zipfile.is_zipfile(file_unduhan):
        raise RuntimeError(f"{tahun} {paruh}: file bukan zip, periksa manual: {file_unduhan}")
    with zipfile.ZipFile(file_unduhan) as z, tempfile.TemporaryDirectory() as tmp:
        z.extractall(tmp)
        for nama in os.listdir(tmp):
            if "oper" in nama:
                shutil.move(os.path.join(tmp, nama), target(tahun, paruh, "angin"))
            elif "wave" in nama:
                shutil.move(os.path.join(tmp, nama), target(tahun, paruh, "gelombang"))
    os.remove(file_unduhan)
    if not sudah_ada(tahun, paruh):
        raise RuntimeError(f"{tahun} {paruh}: isi zip tidak lengkap")


def unduh(tahun, paruh, bulan, percobaan=6):
    permintaan = {
        "product_type": ["reanalysis"],
        "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind",
                     "significant_height_of_combined_wind_waves_and_swell"],
        "year": [str(tahun)],
        "month": [f"{m:02d}" for m in bulan],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": AREA,
    }
    sementara = os.path.join(FOLDER_DATA, f"_unduhan_{tahun}_{paruh}.zip")
    for ke in range(1, percobaan + 1):
        try:
            mulai = time.time()
            cdsapi.Client(quiet=True).retrieve("reanalysis-era5-single-levels", permintaan).download(sementara)
            susun_ulang(sementara, tahun, paruh)
            return f"SELESAI  {tahun} {paruh}  ({(time.time() - mulai) / 60:.0f} menit)"
        except Exception as e:  # antrean penuh, koneksi putus, dsb.
            pesan = str(e).splitlines()[-1] if str(e) else repr(e)
            print(f"  gagal {tahun} {paruh} percobaan {ke}: {pesan}", flush=True)
            # Kalau antrean akun masih penuh, tunggu lebih lama sebelum mencoba lagi
            time.sleep(300 if "queued" in str(e).lower() else 60 * ke)
    return f"GAGAL    {tahun} {paruh}  (jalankan ulang skrip nanti)"


def main():
    os.makedirs(FOLDER_DATA, exist_ok=True)
    tugas = [(t, p, b) for t in TAHUN for p, b in PARUH.items() if not sudah_ada(t, p)]
    print(f"{len(tugas)} permintaan akan dikirim ke Copernicus satu per satu. Biarkan laptop menyala dan terhubung internet.")
    print("Antrean Copernicus bisa memakan beberapa menit sampai beberapa jam per permintaan.\n", flush=True)
    if not tugas:
        print("Semua file sudah lengkap.")
        return
    for i, t in enumerate(tugas, 1):          # satu per satu, sehingga Ctrl+C bisa menghentikan skrip
        print(f"[{i}/{len(tugas)}] mengirim {t[0]} {t[1]} ...", flush=True)
        print(f"[{i}/{len(tugas)}] {unduh(*t)}", flush=True)
    sisa = [(t, p) for t in TAHUN for p in PARUH if not sudah_ada(t, p)]
    print("\nBelum lengkap:" if sisa else "\nSemua data 2016-2025 lengkap.", sisa or "")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nDihentikan. Jalankan ulang perintah yang sama untuk melanjutkan; file yang sudah selesai tidak diunduh ulang.")
