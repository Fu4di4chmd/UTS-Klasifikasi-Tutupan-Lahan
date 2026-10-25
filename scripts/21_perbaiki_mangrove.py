
from pathlib import Path
import numpy as np
import rasterio

BASE = Path(__file__).resolve().parent.parent
MAPS = BASE / "outputs" / "maps"

RF = MAPS / "klasifikasi_rf_jatim_mangrove_19fitur.tif"
KOREKSI = MAPS / "klasifikasi_rf19_waterbody_final_100m.tif"
OUTPUT = MAPS / "klasifikasi_rf19_koreksi_tanpa_mangrove_palsu_100m.tif"

with rasterio.open(RF) as src, rasterio.open(KOREKSI) as corr:
    if (
        src.shape != corr.shape
        or src.crs != corr.crs
        or src.transform != corr.transform
    ):
        raise ValueError("Grid raster tidak sama.")

    asli = src.read(1)
    hasil = corr.read(1)
    profile = corr.profile.copy()

    # Pulihkan semua kelas Mangrove sesuai prediksi RF asli.
    mangrove_asli = asli == 5
    mangrove_palsu = (hasil == 5) & (asli != 5)

    jumlah_palsu = int(mangrove_palsu.sum())

    hasil[mangrove_palsu] = asli[mangrove_palsu]
    hasil[mangrove_asli] = 5

    with rasterio.open(OUTPUT, "w", **profile) as dst:
        dst.write(hasil, 1)
        try:
            dst.write_colormap(1, src.colormap(1))
        except ValueError:
            pass

print("PERBAIKAN SELESAI")
print("Mangrove tambahan yang dibatalkan:", jumlah_palsu)
print("Mangrove RF asli:", int(mangrove_asli.sum()))
print("Mangrove hasil:", int((hasil == 5).sum()))
print("Output:", OUTPUT)
