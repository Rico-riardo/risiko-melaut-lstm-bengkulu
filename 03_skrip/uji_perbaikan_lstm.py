"""Uji perbaikan (Tabel 5 artikel): LSTM dengan masukan spasial (PCA grid ERA5)
dan dua keluaran (nilai angin/gelombang + peluang risiko), dibandingkan dengan
LSTM lokal (fitur titik saja, juga dua keluaran) dan persistence.

Cara pakai (dari folder Data_Artikel):
    python 03_skrip\\uji_perbaikan_lstm.py . 04_hasil\\uji_perbaikan
"""
import os, sys, time, json
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np, pandas as pd, xarray as xr
import tensorflow as tf
from sklearn.decomposition import PCA

AKAR, KELUAR = sys.argv[1], sys.argv[2]
os.makedirs(KELUAR, exist_ok=True)
LOKASI = ["utara", "kota", "kaur"]
FITUR_LOKAL = ["angin_maks", "angin_rata", "u10_rata", "v10_rata", "gelombang_maks", "gelombang_rata", "musim_sin", "musim_cos"]
TARGET = ["angin_maks", "gelombang_maks"]
JENDELA, HORIZON = 14, 3
BA, BG = 15.0, 1.25
LATIH, VALID, UJI = list(range(2016, 2024)), [2024], [2025]
BENIH = [42, 7, 2026]
N_PCA = {"ws_maks": 8, "swh_maks": 8, "u_rata": 4, "v_rata": 4}
KNOT = 1.943844

# ------------------------------------------------------------ fitur spasial harian (WIB)
def grid_harian():
    cache = os.path.join(KELUAR, "grid_harian.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return pd.to_datetime(z["tanggal"]), {k: z[k] for k in N_PCA}
    mentah = os.path.join(AKAR, "01_data_mentah")
    W, G = [], []
    for th in range(2016, 2026):
        for p in ("jan-jun", "jul-des"):
            W.append(xr.open_dataset(f"{mentah}/ERA5_{th}/era5_angin_{th}_{p}.nc")[["u10", "v10"]])
            G.append(xr.open_dataset(f"{mentah}/ERA5_{th}/era5_gelombang_{th}_{p}.nc")[["swh"]])
    w = xr.concat(W, "valid_time"); g = xr.concat(G, "valid_time")
    geser = pd.Timedelta(hours=7)
    w = w.assign_coords(valid_time=w.valid_time + geser); g = g.assign_coords(valid_time=g.valid_time + geser)
    ws = np.hypot(w.u10, w.v10) * KNOT
    hasil = {
        "ws_maks": ws.resample(valid_time="1D").max(),
        "u_rata": w.u10.resample(valid_time="1D").mean(),
        "v_rata": w.v10.resample(valid_time="1D").mean(),
        "swh_maks": g.swh.resample(valid_time="1D").max(),
    }
    n_jam = ws.resample(valid_time="1D").count().isel(latitude=0, longitude=0).values
    lengkap = n_jam == 24
    tanggal = pd.to_datetime(hasil["ws_maks"].valid_time.values)[lengkap]
    arr = {}
    for k, da in hasil.items():
        a = da.values[lengkap].reshape(lengkap.sum(), -1)
        a = a[:, ~np.isnan(a).any(axis=0)]          # buang sel darat (gelombang)
        arr[k] = a.astype("float32")
    np.savez(cache, tanggal=tanggal.values, **arr)
    return tanggal, arr


def fitur_pca(tanggal, arr):
    latih = tanggal.year.isin(LATIH)
    kolom = {}
    for k, a in arr.items():
        mu, sd = a[latih].mean(0), a[latih].std(0) + 1e-6
        z = (a - mu) / sd
        pca = PCA(N_PCA[k], random_state=0).fit(z[latih])
        skor = pca.transform(z)
        print(f"  PCA {k:9s}: {a.shape[1]:3d} sel -> {N_PCA[k]} komponen, varians terjelaskan {pca.explained_variance_ratio_.sum():.1%}")
        for i in range(N_PCA[k]):
            kolom[f"{k}_pc{i+1}"] = skor[:, i]
    return pd.DataFrame(kolom, index=tanggal)


# ------------------------------------------------------------ jendela
def jendela(df, fitur, tahun):
    tgl = df["tanggal"].values; F = df[fitur].values; T = df[TARGET].values; S = df["status_dan"].values
    X, Y, C, H, TG = [], [], [], [], []
    for t in range(JENDELA - 1, len(df) - HORIZON):
        a = t - JENDELA + 1
        if (tgl[t + HORIZON] - tgl[a]) / np.timedelta64(1, "D") != JENDELA + HORIZON - 1:
            continue
        if not np.all(np.isin(pd.DatetimeIndex(tgl[t + 1:t + HORIZON + 1]).year, tahun)):
            continue
        X.append(F[a:t + 1]); Y.append(T[t + 1:t + HORIZON + 1]); C.append(S[t + 1:t + HORIZON + 1])
        H.append(T[t]); TG.append(tgl[t + 1:t + HORIZON + 1])
    return np.array(X, "float32"), np.array(Y), np.array(C), np.array(H), np.array(TG)


def model_dua_keluaran(n_fitur):
    x = tf.keras.Input((JENDELA, n_fitur))
    h = tf.keras.layers.LSTM(64)(x)
    h = tf.keras.layers.Dropout(0.2)(h)
    nilai = tf.keras.layers.Reshape((HORIZON, 2), name="nilai")(tf.keras.layers.Dense(HORIZON * 2)(h))
    peluang = tf.keras.layers.Dense(HORIZON, activation="sigmoid", name="peluang")(h)
    m = tf.keras.Model(x, [nilai, peluang])
    m.compile(tf.keras.optimizers.Adam(1e-3), loss={"nilai": "mse", "peluang": "binary_crossentropy"})
    return m


# ------------------------------------------------------------ metrik
def hitung(a, p):
    tp = int(((a == 1) & (p == 1)).sum()); fn = int(((a == 1) & (p == 0)).sum()); fp = int(((a == 0) & (p == 1)).sum())
    return dict(TP=tp, FN=fn, FP=fp,
                recall=tp / (tp + fn) if tp + fn else np.nan,
                precision=tp / (tp + fp) if tp + fp else np.nan,
                CSI=tp / (tp + fn + fp) if tp + fn + fp else 0.0)

def csi(a, p):
    return hitung(a, p)["CSI"]

def ambang_aturan(Pv, Av):
    best = (-1, BA, BG)
    for ta in np.arange(8, 15.01, 0.25):
        for tg in np.arange(0.8, 1.2501, 0.025):
            c = csi(Av, ((Pv[:, 0] > ta) & (Pv[:, 1] > tg)).astype(int))
            if c > best[0] + 1e-9: best = (c, ta, tg)
    return best[1], best[2]

def ambang_peluang(pv, av):
    best = (-1, 0.5)
    for th in np.arange(0.05, 0.951, 0.01):
        c = csi(av, (pv > th).astype(int))
        if c > best[0] + 1e-9: best = (c, th)
    return best[1]


# ------------------------------------------------------------ utama
def main():
    mulai = time.time()
    harian = pd.read_csv(os.path.join(AKAR, "02_data_olahan", "harian_3lokasi.csv"), parse_dates=["tanggal"])
    doy = harian.tanggal.dt.day_of_year
    harian["musim_sin"] = np.sin(2 * np.pi * doy / 365.25); harian["musim_cos"] = np.cos(2 * np.pi * doy / 365.25)
    print("Membangun fitur spasial harian ...")
    tgl, arr = grid_harian()
    spasial = fitur_pca(tgl, arr)
    FITUR_SPASIAL = list(spasial.columns)

    baris = []
    for lok in LOKASI:
        df = harian[harian.lokasi == lok].sort_values("tanggal").reset_index(drop=True)
        df = df.merge(spasial, left_on="tanggal", right_index=True, how="inner")
        latih = df.tanggal.dt.year.isin(LATIH)
        for varian, fitur in [("LSTM lokal", FITUR_LOKAL), ("LSTM spasial", FITUR_LOKAL + FITUR_SPASIAL)]:
            mu, sd = df.loc[latih, fitur].mean(), df.loc[latih, fitur].std()
            dz = df.copy(); dz[fitur] = (df[fitur] - mu) / sd
            tm, ts = df.loc[latih, TARGET].mean().values, df.loc[latih, TARGET].std().values
            # X dari data terskala; target, status dan persistence dari data asli (satuan knot/m)
            Xl = jendela(dz, fitur, LATIH)[0]; _, Yl, Cl, _, _ = jendela(df, fitur, LATIH)
            Xv = jendela(dz, fitur, VALID)[0]; _, Yv, Cv, Hv, _ = jendela(df, fitur, VALID)
            Xu = jendela(dz, fitur, UJI)[0]; _, Yu, Cu, Hu, _ = jendela(df, fitur, UJI)
            Yl_, Yv_ = (Yl - tm) / ts, (Yv - tm) / ts
            for benih in BENIH:
                tf.keras.utils.set_random_seed(benih)
                m = model_dua_keluaran(len(fitur))
                m.fit(Xl, {"nilai": Yl_, "peluang": Cl}, validation_data=(Xv, {"nilai": Yv_, "peluang": Cv}),
                      epochs=200, batch_size=32, verbose=0,
                      callbacks=[tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)])
                nv, pv = m.predict(Xv, verbose=0); nu, pu = m.predict(Xu, verbose=0)
                nv = nv * ts + tm; nu = nu * ts + tm
                for h in range(HORIZON):
                    rmse_a = float(np.sqrt(np.mean((nu[:, h, 0] - Yu[:, h, 0]) ** 2)))
                    rmse_g = float(np.sqrt(np.mean((nu[:, h, 1] - Yu[:, h, 1]) ** 2)))
                    # (a) nilai ramalan + aturan BMKG asli
                    k = hitung(Cu[:, h], ((nu[:, h, 0] > BA) & (nu[:, h, 1] > BG)).astype(int))
                    baris.append(dict(lokasi=lok, varian=varian, keputusan="aturan BMKG", benih=benih, hari_ke=h + 1, RMSE_angin=rmse_a, RMSE_gel=rmse_g, **k))
                    # (b) nilai ramalan + ambang terkalibrasi (validasi)
                    ta, tg = ambang_aturan(nv[:, h], Cv[:, h])
                    k = hitung(Cu[:, h], ((nu[:, h, 0] > ta) & (nu[:, h, 1] > tg)).astype(int))
                    baris.append(dict(lokasi=lok, varian=varian, keputusan="aturan terkalibrasi", benih=benih, hari_ke=h + 1, RMSE_angin=rmse_a, RMSE_gel=rmse_g, **k))
                    # (c) keluaran peluang + ambang terkalibrasi (validasi)
                    th = ambang_peluang(pv[:, h], Cv[:, h])
                    k = hitung(Cu[:, h], (pu[:, h] > th).astype(int))
                    baris.append(dict(lokasi=lok, varian=varian, keputusan="peluang", benih=benih, hari_ke=h + 1, RMSE_angin=rmse_a, RMSE_gel=rmse_g, **k))
                print(f"  {lok:6s} {varian:13s} benih {benih:4d} selesai ({time.time() - mulai:.0f} dt)")
            if varian == "LSTM lokal":   # persistence sekali saja per lokasi
                for h in range(HORIZON):
                    k = hitung(Cu[:, h], ((Hu[:, 0] > BA) & (Hu[:, 1] > BG)).astype(int))
                    baris.append(dict(lokasi=lok, varian="Persistence", keputusan="aturan BMKG", benih=0, hari_ke=h + 1,
                                      RMSE_angin=float(np.sqrt(np.mean((Hu[:, 0] - Yu[:, h, 0]) ** 2))),
                                      RMSE_gel=float(np.sqrt(np.mean((Hu[:, 1] - Yu[:, h, 1]) ** 2))), **k))
                    ta, tg = ambang_aturan(np.repeat(Hv[:, None, :], 3, 1)[:, h], Cv[:, h])
                    k = hitung(Cu[:, h], ((Hu[:, 0] > ta) & (Hu[:, 1] > tg)).astype(int))
                    baris.append(dict(lokasi=lok, varian="Persistence", keputusan="aturan terkalibrasi", benih=0, hari_ke=h + 1,
                                      RMSE_angin=np.nan, RMSE_gel=np.nan, **k))

    hasil = pd.DataFrame(baris)
    hasil.to_csv(os.path.join(KELUAR, "hasil_per_benih.csv"), index=False)
    ring = (hasil.groupby(["lokasi", "varian", "keputusan", "hari_ke"])
            [["RMSE_angin", "RMSE_gel", "recall", "precision", "CSI"]].agg(["mean", "std"]).round(3))
    ring.to_csv(os.path.join(KELUAR, "ringkasan.csv"))
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)
    tampil = hasil.groupby(["lokasi", "hari_ke", "varian", "keputusan"])[["RMSE_angin", "recall", "precision", "CSI"]].mean().round(3)
    print(tampil.to_string())
    print(f"\nSelesai {(time.time() - mulai) / 60:.1f} menit")


if __name__ == "__main__":
    main()
