
"""
TAHAP 16 - RESAMPLING RF 19 FITUR
HASIL KOREKSI TANPA MANGROVE TAMBAHAN

Input  : Raster koreksi 100 meter
Output : Raster tampilan 50 meter

Metode: Nearest Neighbor

Jalankan:
py -3.9 scripts/16_resampling_peta_50m.py
"""

from pathlib import Path

import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from affine import Affine


# ============================================================
# 1. LOKASI FILE
# ============================================================

BASE = Path(__file__).resolve().parent.parent

INPUT = (
    BASE
    / "outputs"
    / "maps"
    / "klasifikasi_rf19_koreksi_tanpa_mangrove_palsu_100m.tif"
)

OUTPUT = (
    BASE
    / "outputs"
    / "maps"
    / "klasifikasi_rf19_koreksi_tanpa_mangrove_palsu_50m.tif"
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. VALIDASI INPUT
# ============================================================

if not INPUT.exists():
    raise FileNotFoundError(
        f"Raster hasil Tahap 21 tidak ditemukan:\n{INPUT}"
    )

print("=" * 65)
print("RESAMPLING RF 19 FITUR - TANPA MANGROVE TAMBAHAN")
print("=" * 65)


# ============================================================
# 3. RESAMPLING 100 M KE 50 M
# ============================================================

with rasterio.open(INPUT) as src:

    print("Input:", INPUT.name)
    print("CRS:", src.crs)
    print("Resolusi awal:", src.res)
    print("Ukuran awal:", src.width, "x", src.height)

    if src.crs is None:
        raise ValueError("Raster tidak memiliki CRS.")

    if (
        abs(src.res[0] - 100) > 0.01
        or abs(src.res[1] - 100) > 0.01
    ):
        raise ValueError(
            "Raster sumber harus beresolusi 100 meter."
        )

    if src.count != 1:
        raise ValueError(
            "Raster klasifikasi harus memiliki satu band."
        )

    new_width = src.width * 2
    new_height = src.height * 2

    new_transform = src.transform * Affine.scale(0.5, 0.5)

    profile = src.profile.copy()

    profile.update(
        driver="GTiff",
        width=new_width,
        height=new_height,
        transform=new_transform,
        count=1,
        dtype="uint8",
        nodata=0,
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
        BIGTIFF="IF_SAFER"
    )

    print("\nMemulai resampling...")
    print("Ukuran baru:", new_width, "x", new_height)
    print("Resolusi target: 50 meter")

    with rasterio.open(OUTPUT, "w", **profile) as dst:

        reproject(
            source=rasterio.band(src, 1),
            destination=rasterio.band(dst, 1),
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=new_transform,
            dst_crs=src.crs,
            dst_nodata=0,
            resampling=Resampling.nearest,
            num_threads=2
        )

        try:
            dst.write_colormap(1, src.colormap(1))
        except ValueError:
            pass

        dst.set_band_description(
            1,
            "RF 19 fitur - koreksi tanpa mangrove tambahan - grid 50m"
        )


# ============================================================
# 4. VERIFIKASI HASIL
# ============================================================

with rasterio.open(OUTPUT) as check:

    print("\n" + "=" * 65)
    print("RESAMPLING BERHASIL")
    print("=" * 65)

    print("Output:", OUTPUT)
    print("Resolusi output:", check.res)
    print("Ukuran output:", check.width, "x", check.height)
    print("Jumlah band:", check.count)
    print("CRS:", check.crs)

    if (
        check.width != new_width
        or check.height != new_height
    ):
        raise RuntimeError(
            "Ukuran raster output tidak sesuai."
        )

print("\nModel Random Forest tetap menggunakan 19 fitur.")
print("Mangrove tambahan dari koreksi otomatis sudah dibatalkan.")
print("Resolusi informasi asli tetap 100 meter.")
print("Raster dan model asli tidak diubah.")
