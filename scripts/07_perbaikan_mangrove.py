
# ============================================================
# TAHAP 7 - PEMERIKSAAN SEBARAN MANGROVE
# Menggunakan mask zona pesisir 2 km
# ============================================================

from pathlib import Path
import numpy as np
import rasterio

from rasterio.windows import Window
from rasterio.enums import Resampling
from PIL import Image

BASE = Path(__file__).resolve().parent.parent

PETA_ASLI = BASE / "outputs/maps/klasifikasi_rf_jatim.tif"
MASK_PESISIR = BASE / "data/prediksi/mask_pesisir_jatim_100m.tif"

OUTPUT = BASE / "outputs/maps"
OUTPUT.mkdir(parents=True, exist_ok=True)

PETA_HASIL = OUTPUT / "klasifikasi_rf_jatim_pesisir.tif"
PNG_HASIL = OUTPUT / "klasifikasi_rf_jatim_pesisir.png"
MASK_REVIEW = OUTPUT / "mangrove_perlu_review.tif"

# Kelas asli:
# 0 = tidak diklasifikasikan
# 1 = Built-up
# 2 = Farmland
# 3 = Water Body
# 4 = Forest
# 5 = Mangrove

# Kode 6 = prediksi mangrove di luar zona pesisir
#           yang memerlukan pemeriksaan ulang.

WARNA = {
    0: (0, 0, 0, 0),
    1: (128, 128, 128, 255),
    2: (255, 215, 0, 255),
    3: (33, 150, 243, 255),
    4: (34, 139, 34, 255),
    5: (0, 206, 209, 255),
    6: (255, 80, 80, 255)
}

total_mangrove = 0
mangrove_pesisir = 0
mangrove_review = 0

print("=" * 60)
print("PEMERIKSAAN SPASIAL PREDIKSI MANGROVE")
print("=" * 60)

with rasterio.open(PETA_ASLI) as src, \
     rasterio.open(MASK_PESISIR) as pesisir:

    if (
        src.crs != pesisir.crs
        or src.transform != pesisir.transform
        or src.width != pesisir.width
        or src.height != pesisir.height
    ):
        raise ValueError("Grid kedua raster tidak sesuai.")

    profile = src.profile.copy()
    profile.update(
        dtype="uint8",
        nodata=0,
        count=1,
        compress="deflate"
    )

    review_profile = profile.copy()

    with rasterio.open(PETA_HASIL, "w", **profile) as dst, \
         rasterio.open(
             MASK_REVIEW, "w", **review_profile
         ) as review_dst:

        for _, window in src.block_windows(1):

            kelas = src.read(1, window=window)
            zona = pesisir.read(1, window=window)

            mangrove = kelas == 5

            dalam_pesisir = mangrove & (zona == 1)
            luar_pesisir = mangrove & (zona != 1)

            hasil = kelas.copy()
            hasil[luar_pesisir] = 6

            review = np.zeros(
                kelas.shape, dtype=np.uint8
            )
            review[luar_pesisir] = 1

            dst.write(hasil, 1, window=window)
            review_dst.write(review, 1, window=window)

            total_mangrove += np.count_nonzero(mangrove)
            mangrove_pesisir += np.count_nonzero(dalam_pesisir)
            mangrove_review += np.count_nonzero(luar_pesisir)

# ============================================================
# MEMBUAT PREVIEW PNG
# ============================================================

with rasterio.open(PETA_HASIL) as src:

    max_size = 2200
    faktor = max(
        src.width / max_size,
        src.height / max_size,
        1
    )

    lebar = int(src.width / faktor)
    tinggi = int(src.height / faktor)

    data = src.read(
        1,
        out_shape=(tinggi, lebar),
        resampling=Resampling.nearest
    )

rgba = np.zeros((tinggi, lebar, 4), dtype=np.uint8)

for kode, warna in WARNA.items():
    rgba[data == kode] = warna

Image.fromarray(rgba).save(PNG_HASIL)

# ============================================================
# LAPORAN
# ============================================================

print("\nHASIL PEMERIKSAAN")
print("-" * 60)

print(f"Total prediksi Mangrove : {total_mangrove:,}")
print(f"Mangrove zona pesisir   : {mangrove_pesisir:,}")
print(f"Mangrove perlu review   : {mangrove_review:,}")

if total_mangrove > 0:
    persentase = mangrove_review / total_mangrove * 100
    print(f"Persentase perlu review: {persentase:.2f}%")

print("\nFile keluaran:")
print(PETA_HASIL)
print(PNG_HASIL)
print(MASK_REVIEW)

print("\nSelesai. Prediksi asli tidak diubah.")
