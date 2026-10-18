
# ============================================================
# TAHAP 6 - PREDIKSI RANDOM FOREST SELURUH JAWA TIMUR
# Menggunakan model hasil training sebelumnya
# ============================================================

from pathlib import Path
import joblib
import numpy as np
import rasterio

from rasterio.windows import Window
from rasterio.enums import Resampling
from PIL import Image

# ============================================================
# 1. LOKASI DATA
# ============================================================

BASE = Path(__file__).resolve().parent.parent

RASTER_PATH = (
    BASE / "data" / "prediksi"
    / "sentinel2_jatim_14fitur_100m.tif"
)

MODEL_PATH = (
    BASE / "outputs" / "models"
    / "random_forest_model.joblib"
)

OUTPUT_DIR = BASE / "outputs" / "maps"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_TIF = OUTPUT_DIR / "klasifikasi_rf_jatim.tif"
OUTPUT_PNG = OUTPUT_DIR / "klasifikasi_rf_jatim.png"

# ============================================================
# 2. DAFTAR FITUR DAN KELAS
# ============================================================

FEATURES = [
    "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B11", "B12",
    "NDVI", "NDMI", "MNDWI", "NDBI"
]

CLASSES = [
    "build up",
    "farm land",
    "water body",
    "forest",
    "mangrove"
]

# Nilai 0 = tidak diklasifikasikan
KODE_KELAS = {
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
    5: (0, 206, 209, 255)
}

# ============================================================
# 3. LOAD MODEL
# ============================================================

print("=" * 60)
print("PREDIKSI RANDOM FOREST SELURUH JAWA TIMUR")
print("=" * 60)

if not RASTER_PATH.exists():
    raise FileNotFoundError(RASTER_PATH)

if not MODEL_PATH.exists():
    raise FileNotFoundError(MODEL_PATH)

paket = joblib.load(MODEL_PATH)

model = paket["model"]
model_features = list(paket["features"])

if model_features != FEATURES:
    raise ValueError(
        "Urutan fitur model tidak sesuai dengan data prediksi!"
    )

print("Model Random Forest berhasil dimuat.")
print("Jumlah fitur:", len(FEATURES))
print("Kelas model:", list(model.classes_))

if set(model.classes_) != set(CLASSES):
    raise ValueError("Kelas model tidak sesuai.")

# ============================================================
# 4. BACA RASTER DAN SIAPKAN OUTPUT
# ============================================================

with rasterio.open(RASTER_PATH) as src:

    if src.count != 14:
        raise ValueError("Raster harus memiliki 14 band.")

    if list(src.descriptions) != FEATURES:
        raise ValueError(
            "Nama atau urutan band GeoTIFF tidak sesuai!"
        )

    print("Ukuran raster:", src.width, "x", src.height)
    print("CRS:", src.crs)
    print("Resolusi:", src.res)

    profile = src.profile.copy()

    profile.update(
        driver="GTiff",
        count=1,
        dtype="uint8",
        nodata=0,
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256
    )

    jumlah_kelas = np.zeros(6, dtype=np.int64)

    BLOCK_SIZE = 512

    total_blok = (
        ((src.height + BLOCK_SIZE - 1) // BLOCK_SIZE)
        *
        ((src.width + BLOCK_SIZE - 1) // BLOCK_SIZE)
    )

    proses = 0

    # ========================================================
    # 5. PREDIKSI PER BLOK
    # ========================================================

    with rasterio.open(OUTPUT_TIF, "w", **profile) as dst:

        dst.write_colormap(1, WARNA)

        for row in range(0, src.height, BLOCK_SIZE):

            for col in range(0, src.width, BLOCK_SIZE):

                h = min(BLOCK_SIZE, src.height - row)
                w = min(BLOCK_SIZE, src.width - col)

                window = Window(col, row, w, h)

                data = src.read(
                    window=window,
                    out_dtype="float32"
                )

                # Susunan data:
                # (14, tinggi, lebar) -> (jumlah piksel, 14)
                X = np.moveaxis(
                    data, 0, -1
                ).reshape(-1, 14)

                # Filter sesuai preprocessing training
                valid = np.isfinite(X).all(axis=1)
                valid &= (X[:, :10] > 0).all(axis=1)

                # Periksa mask raster jika tersedia
                mask_raster = src.read_masks(
                    1, window=window
                ).reshape(-1) > 0

                valid &= mask_raster

                hasil = np.zeros(
                    X.shape[0],
                    dtype=np.uint8
                )

                if np.any(valid):

                    prediksi = model.predict(X[valid])

                    hasil_valid = np.array(
                        [KODE_KELAS[k] for k in prediksi],
                        dtype=np.uint8
                    )

                    hasil[valid] = hasil_valid

                    jumlah_kelas += np.bincount(
                        hasil_valid,
                        minlength=6
                    )

                hasil = hasil.reshape(h, w)

                dst.write(
                    hasil,
                    1,
                    window=window
                )

                proses += 1

                if proses % 10 == 0 or proses == total_blok:
                    print(
                        f"Blok {proses}/{total_blok} selesai"
                    )

# ============================================================
# 6. SIMPAN PNG UNTUK PREVIEW
# ============================================================

print("\nMembuat gambar preview...")

with rasterio.open(OUTPUT_TIF) as src:

    max_size = 2200

    faktor = max(
        src.width / max_size,
        src.height / max_size,
        1
    )

    new_width = int(src.width / faktor)
    new_height = int(src.height / faktor)

    kelas_png = src.read(
        1,
        out_shape=(new_height, new_width),
        resampling=Resampling.nearest
    )

rgba = np.zeros(
    (new_height, new_width, 4),
    dtype=np.uint8
)

for kode, warna in WARNA.items():
    rgba[kelas_png == kode] = warna

Image.fromarray(rgba, mode="RGBA").save(OUTPUT_PNG)

# ============================================================
# 7. HASIL
# ============================================================

print("\n" + "=" * 60)
print("PREDIKSI SELESAI")
print("=" * 60)

for nama in CLASSES:
    kode = KODE_KELAS[nama]
    print(f"{nama}: {jumlah_kelas[kode]:,} piksel")

print("\nGeoTIFF:", OUTPUT_TIF)
print("Preview PNG:", OUTPUT_PNG)
print("Model yang digunakan: Random Forest lama")
print("Training ulang: Tidak")
