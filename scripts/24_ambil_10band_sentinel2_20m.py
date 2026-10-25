
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import from_bounds, Window
from rasterio.warp import transform_bounds
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from pystac_client import Client
import planetary_computer

BASE = Path(__file__).resolve().parent.parent
OUT_DIR = BASE / "data/prediksi/sentinel2_20m_uji"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT = OUT_DIR / "uji_sentinel2_10band_20m.tif"

BBOX = [112.65, -7.30, 112.85, -7.10]
PERIODE = "2026-06-01/2026-09-30"

BANDS = [
    "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B11", "B12"
]

print("=" * 65)
print("TAHAP 24 REVISI - CARI TILE DENGAN CAKUPAN PENUH")
print("=" * 65)

catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace
)

items = list(catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=BBOX,
    datetime=PERIODE,
    query={"eo:cloud_cover": {"lt": 60}},
    max_items=200
).items())

print("Jumlah kandidat:", len(items))

if not items:
    raise RuntimeError("Tidak ada citra ditemukan.")

kandidat = []

with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):

    # Cek cakupan geometris setiap tile.
    for item in items:

        if not all(b in item.assets for b in BANDS + ["SCL"]):
            continue

        try:
            with rasterio.open(item.assets["B11"].href) as src:
                if src.crs is None:
                    continue

                l, b, r, t = transform_bounds(
                    "EPSG:4326", src.crs, *BBOX
                )

                total_area = (r - l) * (t - b)

                il = max(l, src.bounds.left)
                ib = max(b, src.bounds.bottom)
                ir = min(r, src.bounds.right)
                it = min(t, src.bounds.top)

                if il >= ir or ib >= it:
                    continue

                overlap = (
                    (ir - il) * (it - ib) / total_area
                )

                kandidat.append((
                    overlap,
                    item.properties.get("eo:cloud_cover", 100),
                    item
                ))

        except Exception as exc:
            print("Lewati metadata tile:", str(exc)[:100])

# Utamakan cakupan, kemudian jumlah awan.
kandidat.sort(key=lambda x: (-x[0], x[1]))

if not kandidat:
    raise RuntimeError("Tidak ada tile yang dapat dibaca.")

print("\nLima kandidat terbaik:")
for overlap, cloud, item in kandidat[:5]:
    print(
        item.datetime.date(),
        "| cakupan:", round(overlap * 100, 2), "%",
        "| awan:", cloud, "%"
    )

berhasil = False

for overlap, cloud, item in kandidat:

    # Jangan menyimpan potongan di tepi tile.
    if overlap < 0.80:
        continue

    print("\nMencoba:", item.id)
    print("Cakupan:", round(overlap * 100, 2), "%")
    print("Awan:", cloud, "%")

    try:
        with rasterio.Env(
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"
        ):
            with rasterio.open(item.assets["B11"].href) as ref:
                l, b, r, t = transform_bounds(
                    "EPSG:4326", ref.crs, *BBOX
                )

                window = from_bounds(
                    l, b, r, t, ref.transform
                ).round_offsets().round_lengths()

                window = window.intersection(
                    Window(0, 0, ref.width, ref.height)
                )

                width = int(window.width)
                height = int(window.height)

                transform20 = ref.window_transform(window)
                crs20 = ref.crs

            stack = []

            for name in BANDS:
                with rasterio.open(item.assets[name].href) as src:
                    with WarpedVRT(
                        src,
                        crs=crs20,
                        transform=transform20,
                        width=width,
                        height=height,
                        resampling=Resampling.bilinear,
                        nodata=0
                    ) as vrt:
                        band = vrt.read(1).astype(np.float32)

                stack.append(band)

            with rasterio.open(item.assets["SCL"].href) as src:
                with WarpedVRT(
                    src,
                    crs=crs20,
                    transform=transform20,
                    width=width,
                    height=height,
                    resampling=Resampling.nearest
                ) as vrt:
                    scl = vrt.read(1)

        stack = np.stack(stack)

        valid = (
            np.isfinite(stack).all(axis=0)
            & (stack > 0).all(axis=0)
            & ~np.isin(scl, [0, 1, 3, 8, 9, 10, 11])
        )

        total = valid.size
        jumlah_valid = int(valid.sum())
        persen = 100 * jumlah_valid / total

        print("Ukuran:", width, "x", height)
        print("Piksel valid:", jumlah_valid)
        print("Persentase valid:", round(persen, 2), "%")

        if persen < 70:
            print("Citra dilewati: validitas terlalu rendah.")
            continue

        stack[:, ~valid] = np.nan

        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 10,
            "dtype": "float32",
            "crs": crs20,
            "transform": transform20,
            "nodata": np.nan,
            "compress": "deflate",
            "tiled": True
        }

        with rasterio.open(OUTPUT, "w", **profile) as dst:
            dst.write(stack)

            for i, name in enumerate(BANDS, 1):
                dst.set_band_description(i, name)

            dst.update_tags(
                tanggal=item.datetime.date().isoformat(),
                stac_item=item.id,
                cakupan_persen=round(overlap * 100, 2),
                valid_persen=round(persen, 2),
                tipe_nilai="DN Sentinel-2 L2A"
            )

        print("\nCitra valid berhasil disimpan.")
        berhasil = True
        break

    except Exception as exc:
        print("Gagal:", str(exc)[:250])

if not berhasil:
    raise RuntimeError(
        "Tidak ada tile dengan cakupan >=80% "
        "dan piksel valid >=70% pada kandidat yang diperiksa."
    )

print("\n" + "=" * 65)
print("TAHAP 24 REVISI BERHASIL")
print("=" * 65)
print("Output:", OUTPUT)
print("Persentase valid:", round(persen, 2), "%")
print("Resolusi grid: 20 meter")
