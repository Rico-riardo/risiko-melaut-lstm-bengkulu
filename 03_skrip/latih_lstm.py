"""Ramalan angin dan gelombang harian dengan LSTM, lalu status risiko melaut
menurut kriteria gabungan BMKG untuk perahu nelayan.

Penelitian: Evaluasi Peringatan Dini Risiko Melaut Perahu Nelayan Berbasis
Ramalan LSTM dan Kriteria Gabungan BMKG di Perairan Bengkulu (Rico Riardo).

Cara pakai (setelah praproses_3lokasi.py dijalankan):
    conda activate lstm
    cd /d D:\\ARTIKEL_ILMIAH\\Data_Artikel
    python 03_skrip\\latih_lstm.py

Masukan : 02_data_olahan\\harian_3lokasi.csv   (hanya dibaca)
Keluaran: 04_hasil\\lstm\\...                   (folder baru; data mentah tidak disentuh)

Rancangan:
  - satu model per lokasi (utara, kota, kaur)
  - jendela masukan 14 hari, ramalan 1, 2 dan 3 hari ke depan
  - target: angin maksimum harian (knot) dan gelombang maksimum harian (m)
  - status risiko: ramalan angin > 15 knot DAN gelombang > 1,25 m (BMKG)
  - data latih 2016-2023, validasi 2024, uji 2025
  - pembanding: persistence (besok = hari ini)

Opsi --uji-coba memakai pembagian sementara (latih 2016-2018, validasi 2019,
uji 2024) hanya untuk memeriksa bahwa skrip berjalan sebelum data lengkap.
"""
import os
import sys
import json
import time
import argparse

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")      # sembunyikan log TensorFlow yang tidak perlu

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------------------------------------------------------------- pengaturan
LOKASI = ["utara", "kota", "kaur"]
NAMA_LOKASI = {"utara": "Perairan utara (2,5 LS; 100,5 BT)",
               "kota": "Perairan Kota Bengkulu (4,0 LS; 102,0 BT)",
               "kaur": "Perairan Kaur (5,0 LS; 103,0 BT)"}
FITUR = ["angin_maks", "angin_rata", "u10_rata", "v10_rata", "gelombang_maks", "gelombang_rata",
         "musim_sin", "musim_cos"]
TARGET = ["angin_maks", "gelombang_maks"]
JENDELA = 14            # hari masukan
HORIZON = 3             # hari ke depan
BATAS_ANGIN = 15.0      # knot
BATAS_GELOMBANG = 1.25  # meter
PEMBAGIAN = {"latih": list(range(2016, 2024)), "validasi": [2024], "uji": [2025]}
PEMBAGIAN_UJI_COBA = {"latih": [2016, 2017, 2018], "validasi": [2019], "uji": [2024]}
PARAM = dict(unit_lstm=64, dropout=0.2, laju_belajar=1e-3, batch=32, epoch_maks=200, sabar=20, benih=42)

AKAR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
MASUKAN = os.path.join(AKAR, "02_data_olahan", "harian_3lokasi.csv")
KELUAR = os.path.join(AKAR, "04_hasil", "lstm")


# ---------------------------------------------------------------- data
def siapkan_jendela(df, tahun_target):
    """Pasangan (14 hari masukan, 3 hari target) yang hari targetnya jatuh di tahun_target.
    Hari masukan boleh berasal dari tahun sebelumnya (data itu sudah diketahui saat meramal),
    tetapi seluruh jendela harus berurutan tanpa hari yang hilang."""
    tanggal = df["tanggal"].values
    X, Y, hari_ini, tgl_target = [], [], [], []
    nilai_fitur = df[FITUR].values
    nilai_target = df[TARGET].values
    for t in range(JENDELA - 1, len(df) - HORIZON):
        awal, akhir = t - JENDELA + 1, t + HORIZON
        rentang = (tanggal[akhir] - tanggal[awal]) / np.timedelta64(1, "D")
        if rentang != JENDELA + HORIZON - 1:
            continue                                   # ada hari hilang di dalam jendela
        tahun_hari_target = pd.DatetimeIndex(tanggal[t + 1:t + HORIZON + 1]).year
        if not np.all(np.isin(tahun_hari_target, tahun_target)):
            continue
        X.append(nilai_fitur[awal:t + 1])
        Y.append(nilai_target[t + 1:t + HORIZON + 1])  # bentuk (HORIZON, 2)
        hari_ini.append(nilai_target[t])               # untuk persistence
        tgl_target.append(tanggal[t + 1:t + HORIZON + 1])
    return np.array(X), np.array(Y), np.array(hari_ini), np.array(tgl_target)


def buat_model(n_fitur):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(JENDELA, n_fitur)),
        tf.keras.layers.LSTM(PARAM["unit_lstm"]),
        tf.keras.layers.Dropout(PARAM["dropout"]),
        tf.keras.layers.Dense(HORIZON * len(TARGET)),
        tf.keras.layers.Reshape((HORIZON, len(TARGET))),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(PARAM["laju_belajar"]), loss="mse")
    return model


# ---------------------------------------------------------------- evaluasi
def status_risiko(angin, gelombang):
    return ((angin > BATAS_ANGIN) & (gelombang > BATAS_GELOMBANG)).astype(int)


def metrik_peringatan(aktual, ramalan):
    tp = int(((aktual == 1) & (ramalan == 1)).sum())
    fn = int(((aktual == 1) & (ramalan == 0)).sum())
    fp = int(((aktual == 0) & (ramalan == 1)).sum())
    tn = int(((aktual == 0) & (ramalan == 0)).sum())
    bagi = lambda a, b: round(a / b, 3) if b else float("nan")
    return dict(TP=tp, FN=fn, FP=fp, TN=tn,
                recall=bagi(tp, tp + fn), precision=bagi(tp, tp + fp),
                false_alarm_ratio=bagi(fp, tp + fp), akurasi=bagi(tp + tn, tp + tn + fp + fn),
                CSI=bagi(tp, tp + fn + fp))


def nilai_metrik(lokasi, metode, Y, P, status_aktual):
    regresi, peringatan = [], []
    for h in range(HORIZON):
        baris = dict(lokasi=lokasi, metode=metode, hari_ke=h + 1)
        for j, nama in enumerate(TARGET):
            baris[f"RMSE_{nama}"] = round(float(np.sqrt(mean_squared_error(Y[:, h, j], P[:, h, j]))), 3)
            baris[f"MAE_{nama}"] = round(float(mean_absolute_error(Y[:, h, j], P[:, h, j])), 3)
        regresi.append(baris)
        ramalan = status_risiko(P[:, h, 0], P[:, h, 1])
        peringatan.append(dict(lokasi=lokasi, metode=metode, hari_ke=h + 1,
                               **metrik_peringatan(status_aktual[:, h], ramalan)))
    return regresi, peringatan


# ---------------------------------------------------------------- utama
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uji-coba", action="store_true", help="pembagian sementara sebelum data lengkap")
    args = parser.parse_args()
    bagian = PEMBAGIAN_UJI_COBA if args.uji_coba else PEMBAGIAN
    keluar = KELUAR + ("_uji_coba" if args.uji_coba else "")

    if not os.path.exists(MASUKAN):
        sys.exit("harian_3lokasi.csv belum ada. Jalankan dulu: python 03_skrip\\praproses_3lokasi.py")
    data = pd.read_csv(MASUKAN, parse_dates=["tanggal"])
    hari_dalam_tahun = data["tanggal"].dt.day_of_year
    data["musim_sin"] = np.sin(2 * np.pi * hari_dalam_tahun / 365.25)
    data["musim_cos"] = np.cos(2 * np.pi * hari_dalam_tahun / 365.25)

    tersedia = set(data["tanggal"].dt.year)
    kurang = [t for daftar in bagian.values() for t in daftar if t not in tersedia]
    if kurang:
        sys.exit(f"Tahun berikut belum ada di harian_3lokasi.csv: {kurang}. "
                 "Tunggu unduhan selesai, jalankan praproses lagi, atau pakai --uji-coba.")

    os.makedirs(keluar, exist_ok=True)
    print(f"Pelatihan LSTM  ({'UJI COBA' if args.uji_coba else 'data lengkap'})")
    print("  latih   :", bagian["latih"][0], "-", bagian["latih"][-1])
    print("  validasi:", bagian["validasi"], "  uji:", bagian["uji"])
    print("  keluaran:", keluar, "\n")

    semua_regresi, semua_peringatan, riwayat_semua = [], [], []
    mulai_total = time.time()
    for lokasi in LOKASI:
        tf.keras.utils.set_random_seed(PARAM["benih"])
        df = data[data.lokasi == lokasi].sort_values("tanggal").reset_index(drop=True)

        # Skala fitur dan target memakai statistik data latih saja (mencegah kebocoran data)
        latih_mask = df["tanggal"].dt.year.isin(bagian["latih"])
        rata, simpang = df.loc[latih_mask, FITUR].mean(), df.loc[latih_mask, FITUR].std()
        df_skala = df.copy()
        df_skala[FITUR] = (df[FITUR] - rata) / simpang

        Xl, _, _, _ = siapkan_jendela(df_skala, bagian["latih"])
        Xv, _, _, _ = siapkan_jendela(df_skala, bagian["validasi"])
        Xu, _, _, _ = siapkan_jendela(df_skala, bagian["uji"])
        _, Yl, _, _ = siapkan_jendela(df, bagian["latih"])
        _, Yv, _, _ = siapkan_jendela(df, bagian["validasi"])
        _, Yu, hari_ini_u, tgl_u = siapkan_jendela(df, bagian["uji"])

        t_rata = rata[TARGET].values
        t_simpang = simpang[TARGET].values
        skala_y = lambda Y: (Y - t_rata) / t_simpang

        model = buat_model(len(FITUR))
        henti = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=PARAM["sabar"],
                                                 restore_best_weights=True)
        mulai = time.time()
        riwayat = model.fit(Xl, skala_y(Yl), validation_data=(Xv, skala_y(Yv)),
                            epochs=PARAM["epoch_maks"], batch_size=PARAM["batch"],
                            callbacks=[henti], verbose=0)
        n_epoch = len(riwayat.history["loss"])
        print(f"  {lokasi:6s}: latih {len(Xl)} jendela, validasi {len(Xv)}, uji {len(Xu)} | "
              f"{n_epoch} epoch, {time.time() - mulai:.0f} detik")
        riwayat_semua.append(pd.DataFrame(riwayat.history).assign(lokasi=lokasi, epoch=range(1, n_epoch + 1)))
        model.save(os.path.join(keluar, f"model_{lokasi}.keras"))

        # Ramalan pada data uji
        P_lstm = model.predict(Xu, verbose=0) * t_simpang + t_rata
        P_pers = np.repeat(hari_ini_u[:, None, :], HORIZON, axis=1)          # besok = hari ini
        status_aktual = df.set_index("tanggal")["status_dan"].reindex(tgl_u.ravel()).values.reshape(tgl_u.shape)

        for metode, P in [("LSTM", P_lstm), ("Persistence", P_pers)]:
            r, p = nilai_metrik(lokasi, metode, Yu, P, status_aktual)
            semua_regresi += r
            semua_peringatan += p

        # Simpan ramalan uji (hari ke-1) untuk tabel dan grafik
        tabel = pd.DataFrame({
            "tanggal": tgl_u[:, 0],
            "angin_aktual": Yu[:, 0, 0].round(2), "angin_lstm": P_lstm[:, 0, 0].round(2),
            "angin_persistence": P_pers[:, 0, 0].round(2),
            "gelombang_aktual": Yu[:, 0, 1].round(3), "gelombang_lstm": P_lstm[:, 0, 1].round(3),
            "gelombang_persistence": P_pers[:, 0, 1].round(3),
            "status_aktual": status_aktual[:, 0],
            "status_lstm": status_risiko(P_lstm[:, 0, 0], P_lstm[:, 0, 1]),
            "status_persistence": status_risiko(P_pers[:, 0, 0], P_pers[:, 0, 1]),
        })
        tabel.to_csv(os.path.join(keluar, f"ramalan_uji_{lokasi}.csv"), index=False, date_format="%Y-%m-%d")

        fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
        for a, kol, satuan, batas in [(ax[0], "angin", "knot", BATAS_ANGIN),
                                      (ax[1], "gelombang", "m", BATAS_GELOMBANG)]:
            a.plot(tabel.tanggal, tabel[f"{kol}_aktual"], color="#333333", lw=1.2, label="Aktual (ERA5)")
            a.plot(tabel.tanggal, tabel[f"{kol}_lstm"], color="#1f77b4", lw=1.0, label="Ramalan LSTM (H+1)")
            a.axhline(batas, color="#d62728", ls="--", lw=0.9, label=f"Batas BMKG {batas:g} {satuan}")
            a.set_ylabel(f"{kol.capitalize()} maks. ({satuan})")
            a.grid(alpha=0.3)
        ax[0].legend(loc="upper left", fontsize=8, ncol=3)
        ax[0].set_title(f"{NAMA_LOKASI[lokasi]} - data uji {bagian['uji'][0]}")
        fig.tight_layout()
        fig.savefig(os.path.join(keluar, f"grafik_ramalan_{lokasi}.png"), dpi=200)
        plt.close(fig)

    regresi = pd.DataFrame(semua_regresi)
    peringatan = pd.DataFrame(semua_peringatan)
    regresi.to_csv(os.path.join(keluar, "metrik_regresi.csv"), index=False)
    peringatan.to_csv(os.path.join(keluar, "metrik_peringatan.csv"), index=False)
    pd.concat(riwayat_semua).to_csv(os.path.join(keluar, "riwayat_latih.csv"), index=False)
    with open(os.path.join(keluar, "pengaturan.json"), "w") as f:
        json.dump(dict(pembagian=bagian, fitur=FITUR, target=TARGET, jendela=JENDELA, horizon=HORIZON,
                       batas_angin_knot=BATAS_ANGIN, batas_gelombang_m=BATAS_GELOMBANG,
                       tensorflow=tf.__version__, **PARAM), f, indent=2)

    pd.set_option("display.width", 160)
    print(f"\nSelesai dalam {(time.time() - mulai_total) / 60:.1f} menit.\n")
    print("Ketepatan nilai (RMSE, data uji):")
    print(regresi.pivot_table(index=["lokasi", "hari_ke"], columns="metode",
                              values=["RMSE_angin_maks", "RMSE_gelombang_maks"]).round(3).to_string())
    print("\nKetepatan peringatan (data uji):")
    print(peringatan[["lokasi", "metode", "hari_ke", "TP", "FN", "FP", "recall", "precision",
                      "false_alarm_ratio"]].to_string(index=False))


if __name__ == "__main__":
    main()
