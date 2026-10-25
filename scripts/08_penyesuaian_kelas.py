
from pathlib import Path
import joblib
import numpy as np
import rasterio
from rasterio.enums import Resampling
from PIL import Image

BASE = Path(__file__).resolve().parent.parent

PETA = BASE / "outputs/maps/klasifikasi_rf_jatim.tif"
CITRA = BASE / "data/prediksi/sentinel2_jatim_14fitur_100m.tif"
PESISIR = BASE / "data/prediksi/mask_pesisir_jatim_100m.tif"
MODEL = BASE / "outputs/models/random_forest_model.joblib"

OUTPUT = BASE / "outputs/maps"
HASIL_TIF = OUTPUT / "klasifikasi_rf_jatim_penyesuaian.tif"
HASIL_PNG = OUTPUT / "klasifikasi_rf_jatim_penyesuaian.png"

KODE = {
    "build up": 1,
    "farm land": 2,
    "water body": 3,
    "forest": 4,
    "mangrove": 5
}

WARNA = {
    0: (0, 0, 0, 0),
    1: (128, 128, 128, 255),
    2: (255, 215, 0, 255),
    3: (33, 150, 243, 255),
    4: (34, 139, 34, 255),
    5: (0, 206, 209, 255),
    6: (255, 80, 80, 255)
}

paket = joblib.load(MODEL)
model = paket["model"]

# Urutan kelas mengikuti model yang tersimpan
kelas_model = list(model.classes_)

# Ambang eksperimen untuk menandai prediksi alternatif lemah
AMBANG = 0.35

total_ditinjau = 0
total_diganti = 0
total_tidak_yakin = 0

print("Memulai penyesuaian spasial Mangrove...")

with rasterio.open(PETA) as src, \
     rasterio.open(CITRA) as citra, \
     rasterio.open(PESISIR) as pesisir:

    for raster in [citra, pesisir]:
        if (
            raster.crs != src.crs
            or raster.transform != src.transform
            or raster.width != src.width
            or raster.height != src.height
        ):
            raise ValueError("Grid raster tidak sejajar.")

    if list(citra.descriptions) != list(paket["features"]):
        raise ValueError("Urutan fitur tidak sesuai model.")

    profile = src.profile.copy()
    profile.update(dtype="uint8", nodata=0, compress="deflate")

    with rasterio.open(HASIL_TIF, "w", **profile) as dst:

        for _, window in src.block_windows(1):

            kelas = src.read(1, window=window)
            zona = pesisir.read(1, window=window)

            hasil = kelas.copy()

            # Hanya tinjau prediksi Mangrove di luar pesisir
            target = (kelas == 5) & (zona != 1)

            if np.any(target):
                band = citra.read(window=window).astype("float32")

                X = np.moveaxis(
                    band, 0, -1
                ).reshape(-1, 14)

                posisi = np.flatnonzero(target.ravel())

                X_target = X[posisi]

                valid = np.isfinite(X_target).all(axis=1)
                valid &= (X_target[:, :10] > 0).all(axis=1)

                hasil_flat = hasil.ravel()
                hasil_flat[posisi] = 6

                if np.any(valid):
                    posisi_valid = posisi[valid]

                    proba = model.predict_proba(X_target[valid])

                    # Mangrove tidak menjadi kandidat alternatif
                    proba[:, kelas_model.index("mangrove")] = -1

                    indeks = np.argmax(proba, axis=1)
                    nilai = np.max(proba, axis=1)

                    terima = nilai >= AMBANG

                    alternatif = np.array([
                        KODE[kelas_model[i]] for i in indeks
                    ], dtype=np.uint8)

                    hasil_flat[posisi_valid[terima]] = (
                        alternatif[terima]
                    )

                    total_diganti += int(np.sum(terima))
                    total_tidak_yakin += int(np.sum(~terima))

                total_ditinjau += len(posisi)

            dst.write(hasil, 1, window=window)

print("Membuat PNG preview...")

with rasterio.open(HASIL_TIF) as src:
    faktor = max(src.width / 2200, src.height / 2200, 1)

    w = int(src.width / faktor)
    h = int(src.height / faktor)

    raster = src.read(
        1,
        out_shape=(h, w),
        resampling=Resampling.nearest
    )

rgba = np.zeros((h, w, 4), dtype=np.uint8)

for kode, warna in WARNA.items():
    rgba[raster == kode] = warna

Image.fromarray(rgba).save(HASIL_PNG)

print("\nPENYESUAIAN SELESAI")
print("Piksel ditinjau:", total_ditinjau)
print("Piksel diberi kelas alternatif:", total_diganti)
print("Piksel tetap perlu review:", total_ditinjau - total_diganti)

print("GeoTIFF:", HASIL_TIF)
print("PNG:", HASIL_PNG)
print("Hasil Random Forest asli tetap aman.")
