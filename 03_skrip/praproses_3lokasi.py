"""Praproses data ERA5 untuk tiga titik perairan Bengkulu.

Penelitian: Evaluasi Peringatan Dini Risiko Melaut Perahu Nelayan Berbasis
Ramalan LSTM dan Kriteria Gabungan BMKG di Perairan Bengkulu (Rico Riardo).

Cara pakai (di Anaconda Prompt):
    conda activate lstm
    cd /d D:\\ARTIKEL_ILMIAH\\Data_Artikel
    python 03_skrip\\praproses_3lokasi.py

Masukan : 01_data_mentah\\ERA5_<tahun>\\era5_{angin,gelombang}_<tahun>_{jan-jun,jul-des}.nc
Keluaran: 02_data_olahan\\perjam_3lokasi.csv
          02_data_olahan\\harian_3lokasi.csv
          04_hasil\\ringkasan_praproses.csv

Skrip memproses tahun yang filenya sudah lengkap, jadi bisa dijalankan
sebelum unduhan 2016-2025 selesai dan diulang setelahnya.
"""
import os
import sys
import glob

import numpy as np
import pandas as pd
import xarray as xr

# ---------------------------------------------------------------- pengaturan
LOKASI = {                      # nama: (lintang, bujur), titik grid ERA5 di laut
    "utara": (-2.5, 100.5),     # perairan utara (Mukomuko)
    "kota": (-4.0, 102.0),      # perairan Kota Bengkulu
    "kaur": (-5.0, 103.0),      # perairan Kaur
}
MS_KE_KNOT = 1.943844           # 1 m/s = 1,943844 knot
BATAS_ANGIN = 15.0              # knot  (BMKG, perahu nelayan)
BATAS_GELOMBANG = 1.25          # meter (BMKG, perahu nelayan)
ZONA_WAKTU = 7                  # ERA5 memakai UTC; hari dihitung dalam WIB (UTC+7)
PEMBAGIAN = {"latih": range(2016, 2024), "validasi": [2024], "uji": [2025]}

AKAR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
MENTAH = os.path.join(AKAR, "01_data_mentah")
OLAHAN = os.path.join(AKAR, "02_data_olahan")
HASIL = os.path.join(AKAR, "04_hasil")


def bagian_data(tahun):
    for nama, daftar in PEMBAGIAN.items():
        if tahun in daftar:
            return nama
    return "lain"


def tahun_lengkap():
    """Tahun yang keempat filenya sudah ada."""
    hasil = []
    for folder in sorted(glob.glob(os.path.join(MENTAH, "ERA5_*"))):
        tahun = int(os.path.basename(folder).split("_")[1])
        perlu = [os.path.join(folder, f"era5_{j}_{tahun}_{p}.nc")
                 for j in ("angin", "gelombang") for p in ("jan-jun", "jul-des")]
        if all(os.path.exists(f) for f in perlu):
            hasil.append(tahun)
        else:
            print(f"  lewati {tahun}: file belum lengkap")
    return hasil


def ambil_titik(ds, var, lat, lon):
    """Nilai satu variabel di satu titik grid; berhenti kalau titiknya tidak tepat atau kosong."""
    titik = ds[var].sel(latitude=lat, longitude=lon, method="nearest")
    if abs(float(titik.latitude) - lat) > 1e-6 or abs(float(titik.longitude) - lon) > 1e-6:
        sys.exit(f"Titik {lat},{lon} tidak ada di grid {var}.")
    return titik.values


def baca_tahun(tahun):
    folder = os.path.join(MENTAH, f"ERA5_{tahun}")
    buka = lambda jenis: xr.concat(
        [xr.open_dataset(os.path.join(folder, f"era5_{jenis}_{tahun}_{p}.nc")) for p in ("jan-jun", "jul-des")],
        dim="valid_time")
    angin, gel_ds = buka("angin"), buka("gelombang")
    waktu = pd.to_datetime(angin.valid_time.values)
    if not np.array_equal(angin.valid_time.values, gel_ds.valid_time.values):
        sys.exit(f"{tahun}: waktu file angin dan gelombang tidak sama.")

    tabel = []
    for nama, (lat, lon) in LOKASI.items():
        u = ambil_titik(angin, "u10", lat, lon)
        v = ambil_titik(angin, "v10", lat, lon)
        kec = np.hypot(u, v) * MS_KE_KNOT                    # nilai penuh, belum dibulatkan
        gel = ambil_titik(gel_ds, "swh", lat, lon)
        tabel.append(pd.DataFrame({
            "waktu_utc": waktu,
            "lokasi": nama,
            "u10_ms": u.round(3),
            "v10_ms": v.round(3),
            "angin_knot": kec.round(2),
            "gelombang_m": gel.round(3),
            # status dihitung dari nilai penuh supaya pembulatan tidak menggeser batas 15 knot / 1,25 m
            "berisiko_dan": ((kec > BATAS_ANGIN) & (gel > BATAS_GELOMBANG)).astype(int),
            "berisiko_atau": ((kec > BATAS_ANGIN) | (gel > BATAS_GELOMBANG)).astype(int),
        }))
    angin.close(); gel_ds.close()
    return pd.concat(tabel, ignore_index=True)


def main():
    print("Praproses ERA5 tiga titik perairan Bengkulu")
    daftar = tahun_lengkap()
    if not daftar:
        sys.exit("Belum ada tahun yang lengkap di 01_data_mentah.")
    print("  tahun diproses:", ", ".join(map(str, daftar)))

    perjam = pd.concat([baca_tahun(t) for t in daftar], ignore_index=True)
    perjam = perjam.sort_values(["lokasi", "waktu_utc"]).reset_index(drop=True)

    # Pemeriksaan mutu: jam hilang, jam dobel, nilai kosong
    masalah = []
    for nama, g in perjam.groupby("lokasi"):
        selisih = g["waktu_utc"].diff().dropna().dt.total_seconds().div(3600)
        if g["waktu_utc"].duplicated().any():
            masalah.append(f"{nama}: ada jam dobel")
        celah = selisih[selisih > 1]
        if len(celah):
            masalah.append(f"{nama}: {len(celah)} celah waktu (tahun yang belum diunduh ikut terhitung)")
        if g[["angin_knot", "gelombang_m"]].isna().any().any():
            masalah.append(f"{nama}: ada nilai kosong")

    perjam["waktu_wib"] = perjam["waktu_utc"] + pd.Timedelta(hours=ZONA_WAKTU)

    # Ringkasan harian (hari WIB, hanya hari yang lengkap 24 jam)
    perjam["tanggal"] = perjam["waktu_wib"].dt.normalize()
    harian = (perjam.groupby(["lokasi", "tanggal"])
              .agg(jumlah_jam=("angin_knot", "size"),
                   angin_maks=("angin_knot", "max"),
                   angin_rata=("angin_knot", "mean"),
                   u10_rata=("u10_ms", "mean"),
                   v10_rata=("v10_ms", "mean"),
                   gelombang_maks=("gelombang_m", "max"),
                   gelombang_rata=("gelombang_m", "mean"),
                   jam_berisiko=("berisiko_dan", "sum"),
                   jam_berisiko_atau=("berisiko_atau", "sum"))
              .reset_index())
    tidak_lengkap = (harian.jumlah_jam < 24).sum()
    harian = harian[harian.jumlah_jam == 24].drop(columns="jumlah_jam")
    harian["status_dan"] = (harian.jam_berisiko > 0).astype(int)        # aturan BMKG (angin DAN gelombang)
    harian["status_atau"] = (harian.jam_berisiko_atau > 0).astype(int)  # pembanding (angin ATAU gelombang)
    harian["tahun"] = harian.tanggal.dt.year
    harian["bagian"] = harian.tahun.map(bagian_data)
    for kol in ["angin_maks", "angin_rata", "gelombang_maks", "gelombang_rata"]:
        harian[kol] = harian[kol].round(2)
    for kol in ["u10_rata", "v10_rata"]:
        harian[kol] = harian[kol].round(3)

    os.makedirs(OLAHAN, exist_ok=True)
    os.makedirs(HASIL, exist_ok=True)
    perjam.drop(columns="tanggal").to_csv(os.path.join(OLAHAN, "perjam_3lokasi.csv"), index=False)
    harian.to_csv(os.path.join(OLAHAN, "harian_3lokasi.csv"), index=False, date_format="%Y-%m-%d")

    ringkas = (harian.groupby(["tahun", "lokasi"])
               .agg(hari=("status_dan", "size"), hari_berisiko=("status_dan", "sum"),
                    hari_berisiko_atau=("status_atau", "sum"),
                    angin_maks=("angin_maks", "max"), gelombang_maks=("gelombang_maks", "max"))
               .reset_index())
    ringkas.to_csv(os.path.join(HASIL, "ringkasan_praproses.csv"), index=False)

    print(f"\n  data per jam : {len(perjam):,} baris  -> 02_data_olahan\\perjam_3lokasi.csv")
    print(f"  data harian  : {len(harian):,} baris  -> 02_data_olahan\\harian_3lokasi.csv")
    print(f"  hari tepi yang dibuang (kurang dari 24 jam WIB): {tidak_lengkap}")
    print("\n  Hari berisiko (aturan DAN) per tahun:")
    tabel = ringkas.pivot(index="tahun", columns="lokasi", values="hari_berisiko")[list(LOKASI)]
    print(tabel.to_string())
    print("\n  Pemeriksaan mutu:", "tidak ada masalah" if not masalah else "")
    for m in masalah:
        print("   -", m)


if __name__ == "__main__":
    main()
