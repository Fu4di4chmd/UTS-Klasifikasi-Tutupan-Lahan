
"""
TAHAP 25 - MEMBUAT 19 FITUR SENTINEL-2 GRID 20 M
Input: 10 band DN dari Tahap 24
Output: 19 fitur float32

Hanya untuk uji satu citra tahun 2026.
Belum digunakan untuk training atau prediksi provinsi.
"""

from pathlib import Path
import numpy as np
import rasterio

BASE = Path(__file__).resolve().parent.parent

FOLDER = BASE / "data" / "prediksi" / "sentinel2_20m_uji"

INPUT = FOLDER / "uji_sentinel2_10band_20m.tif"
OUTPUT = FOLDER / "uji_sentinel2_19fitur_20m.tif"

BANDS = [
    "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B11", "B12"
]

FEATURES = BANDS + [
    "NDVI", "NDMI", "MNDWI", "NDBI",
    "NDWI", "MVI", "REMI",
    "MANGROVE_8A_11", "VEG_8A_4"
]

EPS = 1e-6


def ratio(n, d, absolute=False):
    n = np.asarray(n, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)

    if absolute:
        n = np.abs(n)
        d = np.abs(d)

    result = np.full(n.shape, np.nan, dtype=np.float64)

    valid = (
        np.isfinite(n)
        & np.isfinite(d)
        & (np.abs(d) > EPS)
    )

    np.divide(n, d, out=result, where=valid)

    return result.astype(np.float32)


if not INPUT.exists():
    raise FileNotFoundError(INPUT)

print("=" * 65)
print("TAHAP 25 - SENTINEL-2 19 FITUR GRID 20 M")
print("=" * 65)

with rasterio.open(INPUT) as src:

    if src.count != 10 or list(src.descriptions) != BANDS:
        raise ValueError("Urutan 10 band tidak sesuai.")

    dn = src.read().astype(np.float32)

    # Mask invalid dari Tahap 24.
    valid = np.isfinite(dn).all(axis=0)
    valid &= (dn > 0).all(axis=0)

    # Koreksi offset L2A baseline 04.00 ke atas.
    refl = (dn - 1000.0) / 10000.0

    refl[:, ~valid] = np.nan

    b = {
        name: refl[i]
        for i, name in enumerate(BANDS)
    }

    indices = [
        ratio(b["B08"] - b["B04"],
              b["B08"] + b["B04"]),

        ratio(b["B08"] - b["B11"],
              b["B08"] + b["B11"]),

        ratio(b["B03"] - b["B11"],
              b["B03"] + b["B11"]),

        ratio(b["B11"] - b["B08"],
              b["B11"] + b["B08"]),

        ratio(b["B03"] - b["B08"],
              b["B03"] + b["B08"]),

        ratio(b["B08"] - b["B03"],
              b["B11"] - b["B03"],
              absolute=True),

        ratio(b["B05"] - b["B04"],
              b["B11"] - b["B03"]),

        ratio(b["B8A"] - b["B11"],
              b["B8A"] + b["B11"]),

        ratio(b["B8A"] - b["B04"],
              b["B8A"] + b["B04"])
    ]

    stack = np.concatenate(
        [refl, np.stack(indices)],
        axis=0
    ).astype(np.float32)

    # Pastikan semua indeks dihitung hanya dari piksel valid.
    stack[:, ~valid] = np.nan

    profile = src.profile.copy()
    profile.update(
        driver="GTiff",
        count=19,
        dtype="float32",
        nodata=np.nan,
        compress="deflate",
        tiled=True
    )

    with rasterio.open(OUTPUT, "w", **profile) as dst:

        dst.write(stack)

        for i, name in enumerate(FEATURES, 1):
            dst.set_band_description(i, name)

        dst.update_tags(
            sumber="Sentinel-2 L2A via Microsoft Planetary Computer",
            keterangan="Uji satu citra 2026",
            resolusi_grid="20 meter",
            reflektansi="(DN - 1000) / 10000",
            jumlah_fitur="19"
        )

jumlah_valid = int(
    np.isfinite(stack).all(axis=0).sum()
)

print("\n" + "=" * 65)
print("TAHAP 25 BERHASIL")
print("=" * 65)

print("Jumlah fitur:", stack.shape[0])
print("Ukuran:", stack.shape[2], "x", stack.shape[1])
print("Piksel valid 19 fitur:", jumlah_valid)
print("Output:", OUTPUT)
print("Belum dilakukan training atau prediksi.")
