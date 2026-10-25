
"""Audit dan koreksi konservatif Water Body pada raster 100 meter.
Tidak mengubah model, GeoJSON, maupun raster klasifikasi asli.
"""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy.ndimage import label, convolve

BASE = Path(__file__).resolve().parent.parent

RF = BASE / "outputs/maps/klasifikasi_rf_jatim_mangrove_17fitur.tif"
S2 = BASE / "data/prediksi/sentinel2_jatim_14fitur_100m.tif"

OUT = BASE / "outputs/maps"
OUT.mkdir(parents=True, exist_ok=True)

AUDIT = OUT / "audit_waterbody_17fitur_100m.tif"
CLEAN = OUT / "klasifikasi_rf_jatim_17fitur_waterbody_koreksi_100m.tif"

# Aturan koreksi konservatif
MAX_PATCH = 2
MAX_NDWI = -0.10
MAX_MNDWI = -0.10

for fp in (RF, S2):
    if not fp.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {fp}")

print("Membaca raster Random Forest dan Sentinel-2...")

with rasterio.open(RF) as src:
    classified = src.read(1)
    profile = src.profile.copy()
    transform = src.transform
    crs = src.crs
    height = src.height
    width = src.width

with rasterio.open(S2) as src:

    if src.count < 13:
        raise ValueError("Raster Sentinel-2 tidak lengkap.")

    # Susunan band:
    # 2=B03, 7=B08, 13=MNDWI

    def aligned_band(idx):
        a = np.full((height, width), np.nan, np.float32)

        reproject(
            source=rasterio.band(src, idx),
            destination=a,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=transform,
            dst_crs=crs,
            dst_nodata=np.nan,
            resampling=Resampling.nearest
        )
        return a

    green = aligned_band(2)
    nir = aligned_band(7)
    mndwi = aligned_band(13)

# Menghitung NDWI
with np.errstate(divide="ignore", invalid="ignore"):
    ndwi = (green - nir) / (green + nir)

valid_index = (
    np.isfinite(ndwi)
    & np.isfinite(mndwi)
    & (np.abs(green + nir) > 1e-8)
)

del green, nir

# Kode kelas Water Body adalah 3
water = classified == 3

# Identifikasi kelompok Water Body
components, n = label(
    water,
    structure=np.ones((3, 3), dtype=np.uint8)
)

sizes = np.bincount(components.ravel())

# Kelompok kecil: maksimal 2 piksel asli
small = water & (sizes[components] <= MAX_PATCH)

# Indikasi spektral kurang mendukung air
weak = (
    valid_index
    & (ndwi < MAX_NDWI)
    & (mndwi < MAX_MNDWI)
)

candidate = small & weak

# Periksa delapan piksel tetangga
kernel = np.ones((3, 3), np.uint8)
kernel[1, 1] = 0

neighbor_counts = np.zeros(
    (4, height, width),
    dtype=np.uint8
)

classes = [1, 2, 4, 5]

for j, klass in enumerate(classes):
    neighbor_counts[j] = convolve(
        (classified == klass).astype(np.uint8),
        kernel,
        mode="constant",
        cval=0
    )

best_idx = np.argmax(neighbor_counts, axis=0)
best_n = np.max(neighbor_counts, axis=0)

# Koreksi jika minimal 5 tetangga sepakat
change = candidate & (best_n >= 5)

replacement = np.array(classes, dtype=np.uint8)[best_idx]

corrected = classified.copy()
corrected[change] = replacement[change]

# Membuat raster audit
# 0 = bukan Water Body
# 1 = Water Body dipertahankan
# 2 = kandidat mencurigakan
# 3 = piksel yang dikoreksi

audit = np.zeros(classified.shape, dtype=np.uint8)
audit[water] = 1
audit[candidate] = 2
audit[change] = 3

profile.update(
    count=1,
    dtype="uint8",
    compress="deflate",
    tiled=True,
    blockxsize=256,
    blockysize=256
)

# Simpan audit
with rasterio.open(AUDIT, "w", **profile) as dst:
    dst.write(audit, 1)
    dst.set_band_description(1, "Audit Water Body")

# Simpan raster koreksi
with rasterio.open(CLEAN, "w", **profile) as dst:
    dst.write(corrected, 1)
    dst.set_band_description(
        1,
        "Random Forest 17 fitur - koreksi terbatas Water Body"
    )

print("\n===== HASIL AUDIT WATER BODY =====")

print("Jumlah kelompok Water Body:", n)
print("Water Body awal:", int(water.sum()))
print("Piksel kelompok kecil:", int(small.sum()))
print("Kandidat dengan indeks air negatif:", int(candidate.sum()))
print("Piksel yang dikoreksi:", int(change.sum()))
print("Water Body setelah koreksi:", int((corrected == 3).sum()))

print("\nRaster audit:", AUDIT)
print("Raster koreksi:", CLEAN)
print("\nSELESAI")
print("File asli dan model Random Forest tidak diubah.")
print("Koreksi ini bersifat percobaan dan masih perlu divalidasi.")
