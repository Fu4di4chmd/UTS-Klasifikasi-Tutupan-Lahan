
"""
TAHAP 19 - SCREENING WATER BODY
Random Forest 19 Fitur + Dynamic World

Tujuan:
- Memeriksa prediksi Water Body yang kurang didukung referensi.
- Menandai piksel tersebut sebagai NoData pada raster percobaan.
- Tidak mengubah model RF atau raster aslinya.

Jalankan:
py -3.9 scripts/19_filter_waterbody_dw.py
"""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling


# ============================================================
# 1. LOKASI FILE
# ============================================================

BASE = Path(__file__).resolve().parent.parent

RF = (
    BASE / "outputs" / "maps"
    / "klasifikasi_rf_jatim_mangrove_19fitur.tif"
)

DW = (
    BASE / "data" / "prediksi"
    / "referensi_water_jatim_100m_2026.tif"
)

OUTPUT = (
    BASE / "outputs" / "maps"
    / "klasifikasi_rf19_waterbody_screening_100m.tif"
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

AMBANG_AIR = 20


# ============================================================
# 2. VALIDASI FILE
# ============================================================

for path in [RF, DW]:
    if not path.exists():
        raise FileNotFoundError(
            f"File tidak ditemukan:\n{path}"
        )


# ============================================================
# 3. MEMBACA DAN MEMPROSES RASTER
# ============================================================

print("=" * 65)
print("SCREENING WATER BODY - RF 19 FITUR + DYNAMIC WORLD")
print("=" * 65)

total_water_awal = 0
total_ditandai = 0
total_water_akhir = 0

with rasterio.open(RF) as src, rasterio.open(DW) as ref:

    print("Raster RF:", RF.name)
    print("Referensi DW:", DW.name)
    print("Resolusi RF:", src.res)
    print("Resolusi DW:", ref.res)

    profile = src.profile.copy()

    profile.update(
        driver="GTiff",
        count=1,
        dtype="uint8",
        nodata=0,
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
        BIGTIFF="IF_SAFER"
    )

    with WarpedVRT(
        ref,
        crs=src.crs,
        transform=src.transform,
        width=src.width,
        height=src.height,
        resampling=Resampling.nearest,
        src_nodata=255,
        nodata=255
    ) as dw_aligned:

        with rasterio.open(OUTPUT, "w", **profile) as dst:

            try:
                dst.write_colormap(1, src.colormap(1))
            except ValueError:
                pass

            dst.set_band_description(
                1,
                "RF 19 fitur - screening Water Body Dynamic World"
            )

            for _, window in src.block_windows(1):

                rf = src.read(1, window=window)
                dw = dw_aligned.read(1, window=window)

                water_awal = rf == 3

                # Hanya piksel RF Water Body yang memiliki
                # referensi Dynamic World valid dan <20%.
                kandidat = (
                    water_awal
                    & (dw != 255)
                    & (dw < AMBANG_AIR)
                )

                hasil = rf.copy()

                # 0 = NoData / perlu pemeriksaan
                # BUKAN otomatis berubah menjadi daratan.
                hasil[kandidat] = 0

                total_water_awal += int(water_awal.sum())
                total_ditandai += int(kandidat.sum())
                total_water_akhir += int((hasil == 3).sum())

                dst.write(hasil, 1, window=window)


# ============================================================
# 4. HASIL
# ============================================================

print("\n" + "=" * 65)
print("SCREENING SELESAI")
print("=" * 65)

print(f"Water Body awal       : {total_water_awal:,} piksel")
print(f"Piksel ditandai       : {total_ditandai:,} piksel")
print(f"Water Body tersisa    : {total_water_akhir:,} piksel")

print("\nOutput:", OUTPUT)

print("\nCATATAN:")
print("- Model Random Forest 19 fitur tidak diubah.")
print("- Raster klasifikasi asli tidak ditimpa.")
print("- Piksel yang ditandai menjadi NoData, bukan kelas lain.")
print("- Hasil ini masih berupa screening, bukan klasifikasi final.")
