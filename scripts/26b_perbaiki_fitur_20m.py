
from pathlib import Path
import numpy as np
import rasterio

BASE = Path(__file__).resolve().parent.parent
FOLDER = BASE / "data/prediksi/sentinel2_20m_uji"

INPUT = FOLDER / "uji_sentinel2_10band_20m.tif"
OUTPUT = FOLDER / "uji_sentinel2_19fitur_bersih_20m.tif"

BANDS = [
    "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B11", "B12"
]

FEATURES = BANDS + [
    "NDVI", "NDMI", "MNDWI", "NDBI",
    "NDWI", "MVI", "REMI",
    "MANGROVE_8A_11", "VEG_8A_4"
]


def rasio(a, b, minimum=1e-6, absolut=False):
    if absolut:
        a = np.abs(a)
        b = np.abs(b)

    hasil = np.full(
        a.shape, np.nan, dtype=np.float32
    )

    valid = (
        np.isfinite(a)
        & np.isfinite(b)
        & (np.abs(b) >= minimum)
    )

    np.divide(a, b, out=hasil, where=valid)

    return hasil


print("=" * 65)
print("TAHAP 26B - PERBAIKAN 19 FITUR GRID 20 M")
print("=" * 65)

with rasterio.open(INPUT) as src:
    if src.count != 10:
        raise ValueError("Raster harus memiliki 10 band.")

    if list(src.descriptions) != BANDS:
        raise ValueError("Urutan band tidak sesuai.")

    dn = src.read().astype(np.float32)
    profile = src.profile.copy()

    valid_asli = (
        np.isfinite(dn).all(axis=0)
        & (dn > 0).all(axis=0)
    )

    # Offset dan kuantifikasi telah diverifikasi Tahap 26.
    refl = (dn - 1000.0) / 10000.0

    # Harmonisasi reflektansi negatif untuk analisis.
    refl = np.maximum(refl, 0.0)

    refl[:, ~valid_asli] = np.nan

    b = {name: refl[i] for i, name in enumerate(BANDS)}

    indices = [
        rasio(
            b["B08"] - b["B04"],
            b["B08"] + b["B04"]
        ),
        rasio(
            b["B08"] - b["B11"],
            b["B08"] + b["B11"]
        ),
        rasio(
            b["B03"] - b["B11"],
            b["B03"] + b["B11"]
        ),
        rasio(
            b["B11"] - b["B08"],
            b["B11"] + b["B08"]
        ),
        rasio(
            b["B03"] - b["B08"],
            b["B03"] + b["B08"]
        ),
        # MVI dan REMI memakai syarat pembagi lebih
        # besar agar tidak menghasilkan nilai ekstrem
        # akibat pembagi mendekati nol.
        rasio(
            b["B08"] - b["B03"],
            b["B11"] - b["B03"],
            minimum=0.02,
            absolut=True
        ),
        rasio(
            b["B05"] - b["B04"],
            b["B11"] - b["B03"],
            minimum=0.02
        ),
        rasio(
            b["B8A"] - b["B11"],
            b["B8A"] + b["B11"]
        ),
        rasio(
            b["B8A"] - b["B04"],
            b["B8A"] + b["B04"]
        )
    ]

    stack = np.concatenate(
        [refl, np.stack(indices)],
        axis=0
    ).astype(np.float32)

    valid19 = np.isfinite(stack).all(axis=0)

    # Semua fitur harus tersedia pada piksel yang sama.
    stack[:, ~valid19] = np.nan

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

        for i, nama in enumerate(FEATURES, 1):
            dst.set_band_description(i, nama)

        dst.update_tags(
            reflektansi="max((DN-1000)/10000, 0)",
            minimum_pembagi_mvi_remi="0.02",
            status="uji prapemrosesan 20 meter"
        )

total = valid19.size

print("\nHASIL:")
print("Total piksel:", total)
print("Valid semula:", int(valid_asli.sum()))
print("Valid setelah koreksi:", int(valid19.sum()))
print(
    "Persentase valid:",
    round(100 * valid19.sum() / total, 2),
    "%"
)

print("\nRentang indeks hasil:")

for i, nama in enumerate(FEATURES[10:], 10):
    nilai = stack[i]
    finite = nilai[np.isfinite(nilai)]

    if finite.size:
        print(
            nama,
            "min:", round(float(finite.min()), 4),
            "median:", round(float(np.median(finite)), 4),
            "max:", round(float(finite.max()), 4)
        )

print("\nOutput:", OUTPUT)
print("File lama tidak diubah.")
print("Selesai.")
