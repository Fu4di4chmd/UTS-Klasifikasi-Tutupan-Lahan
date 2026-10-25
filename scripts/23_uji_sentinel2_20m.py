
"""
TAHAP 23 - UJI SENTINEL-2 ASLI 20 METER
Tanpa Google Earth Engine.

Mengambil satu potongan band B11 dari
Microsoft Planetary Computer.

Tidak mengubah model atau raster lama.
"""

from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.windows import Window
from rasterio.warp import transform_bounds

from pystac_client import Client
import planetary_computer


BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "data" / "prediksi" / "sentinel2_20m_uji"
OUT.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT / "uji_B11_asli_20m.tif"

# Wilayah uji pesisir Surabaya-Sidoarjo.
# Format: barat, selatan, timur, utara (longitude/latitude).
BBOX = [112.65, -7.30, 112.85, -7.10]

TANGGAL = "2026-06-01/2026-09-30"

print("=" * 60)
print("UJI DATA SENTINEL-2 ASLI 20 METER")
print("=" * 60)

print("Menghubungkan ke Planetary Computer...")

catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace
)

print("Mencari citra Sentinel-2...")

search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=BBOX,
    datetime=TANGGAL,
    query={
        "eo:cloud_cover": {"lt": 60}
    },
    max_items=60
)

items = list(search.items())

if not items:
    raise RuntimeError(
        "Tidak ditemukan citra. Coba ubah periode tanggal."
    )

# Utamakan persentase awan rendah.
items.sort(
    key=lambda item: item.properties.get(
        "eo:cloud_cover", 100
    )
)

print("Jumlah citra ditemukan:", len(items))

berhasil = False

for nomor, item in enumerate(items, start=1):

    if "B11" not in item.assets:
        continue

    print("\nMencoba citra:", nomor)
    print("Tanggal:", item.datetime.date())
    print(
        "Awan:",
        item.properties.get("eo:cloud_cover"),
        "%"
    )

    try:
        url = item.assets["B11"].href

        # Baca sebagian COG melalui internet.
        with rasterio.Env(
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
            CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF",
        ):
            with rasterio.open(url) as src:

                print("CRS:", src.crs)
                print("Resolusi asli:", src.res)

                if (
                    abs(src.res[0] - 20) > 0.1
                    or abs(src.res[1] - 20) > 0.1
                ):
                    raise ValueError(
                        "Band B11 bukan resolusi asli 20 meter."
                    )

                # Ubah batas wilayah ke CRS citra.
                left, bottom, right, top = transform_bounds(
                    "EPSG:4326",
                    src.crs,
                    *BBOX
                )

                # Potong batas dengan cakupan tile.
                left = max(left, src.bounds.left)
                right = min(right, src.bounds.right)
                bottom = max(bottom, src.bounds.bottom)
                top = min(top, src.bounds.top)

                if left >= right or bottom >= top:
                    print("Tile tidak mencakup area uji.")
                    continue

                window = from_bounds(
                    left, bottom, right, top,
                    transform=src.transform
                )

                window = window.round_offsets().round_lengths()

                window = window.intersection(
                    Window(0, 0, src.width, src.height)
                )

                if window.width <= 0 or window.height <= 0:
                    continue

                print(
                    "Mengambil potongan:",
                    int(window.width),
                    "x",
                    int(window.height),
                    "piksel"
                )

                data = src.read(1, window=window)

                profile = src.profile.copy()
                profile.update(
                    driver="GTiff",
                    height=data.shape[0],
                    width=data.shape[1],
                    transform=src.window_transform(window),
                    count=1,
                    compress="deflate",
                    tiled=True
                )

                with rasterio.open(
                    OUTPUT, "w", **profile
                ) as dst:
                    dst.write(data, 1)
                    dst.set_band_description(
                        1, "Sentinel-2 B11 asli 20 meter"
                    )

                print("Citra disimpan:", OUTPUT)
                print("Nilai piksel minimum:", int(np.min(data)))
                print("Nilai piksel maksimum:", int(np.max(data)))

                berhasil = True
                break

    except Exception as exc:
        print("Citra ini gagal:", str(exc)[:250])
        continue

if not berhasil:
    raise RuntimeError(
        "Belum berhasil membaca citra. "
        "Periksa koneksi internet atau akses data."
    )

print("\n" + "=" * 60)
print("UJI SENTINEL-2 20 METER BERHASIL")
print("=" * 60)

with rasterio.open(OUTPUT) as hasil:
    print("Output:", OUTPUT)
    print("Resolusi:", hasil.res)
    print("Ukuran:", hasil.width, "x", hasil.height)
    print("CRS:", hasil.crs)

print("Data ini merupakan band asli 20 meter.")
print("Model RF dan semua file lama tetap aman.")
