
"""
TAHAP 22 - PIPELINE PEMULIHAN SAWAH DAN PETA INTERAKTIF
Random Forest 19 fitur + koreksi eksperimental

Proses:
1. Memulihkan Farm Land hasil RF asli.
2. Memulihkan NoData yang disebabkan koreksi sebelumnya.
3. Mempertahankan hanya Mangrove asli RF.
4. Resampling tampilan 100 m menjadi 50 m.
5. Membuat peta interaktif HTML.

Jalankan:
py -3.9 scripts/22_pipeline_peta_sawah.py
"""

from pathlib import Path
from io import BytesIO
import base64
import json

import numpy as np
import rasterio
import folium

from PIL import Image
from affine import Affine
from rasterio.enums import Resampling
from rasterio.warp import reproject, calculate_default_transform


# ============================================================
# 1. LOKASI FILE
# ============================================================

BASE = Path(__file__).resolve().parent.parent
MAPS = BASE / "outputs" / "maps"
GEOJSON_DIR = BASE / "data" / "geojson"

MAPS.mkdir(parents=True, exist_ok=True)

RF_ASLI = MAPS / "klasifikasi_rf_jatim_mangrove_19fitur.tif"

KOREKSI_LAMA = MAPS / (
    "klasifikasi_rf19_koreksi_tanpa_mangrove_palsu_100m.tif"
)

OUTPUT_100M = MAPS / "klasifikasi_rf19_sawah_dipulihkan_100m.tif"

OUTPUT_50M = MAPS / "klasifikasi_rf19_sawah_dipulihkan_50m.tif"

OUTPUT_HTML = MAPS / "peta_interaktif_rf19_sawah_dipulihkan.html"


# ============================================================
# 2. KELAS DAN WARNA
# ============================================================

KELAS = {
    1: ("Built-up", "#d45b56", "build up"),
    2: ("Farm land", "#edaa63", "farm land"),
    3: ("Water body", "#298fe8", "water body"),
    4: ("Forest", "#208653", "forest"),
    5: ("Mangrove", "#04bfb4", "mangrove"),
}


def data_url(array):
    buffer = BytesIO()
    Image.fromarray(array).save(
        buffer,
        format="PNG",
        optimize=True
    )
    return (
        "data:image/png;base64,"
        + base64.b64encode(buffer.getvalue()).decode("ascii")
    )


# ============================================================
# 3. VALIDASI
# ============================================================

print("=" * 65)
print("PIPELINE PEMULIHAN SAWAH - RANDOM FOREST 19 FITUR")
print("=" * 65)

for path in [RF_ASLI, KOREKSI_LAMA]:
    if not path.exists():
        raise FileNotFoundError(
            f"File input tidak ditemukan:\n{path}"
        )


# ============================================================
# 4. MEMULIHKAN SAWAH DAN MANGROVE ASLI
# ============================================================

print("\n[1/3] Memulihkan klasifikasi...")

with rasterio.open(RF_ASLI) as src, rasterio.open(KOREKSI_LAMA) as corr:

    if (
        src.shape != corr.shape
        or src.crs != corr.crs
        or src.transform != corr.transform
    ):
        raise ValueError("Grid raster asli dan koreksi berbeda.")

    asli = src.read(1)
    hasil = corr.read(1).copy()

    profile = src.profile.copy()

    # Pulihkan piksel NoData buatan yang masih
    # mempunyai prediksi kelas pada raster RF asli.
    kosong_koreksi = (
        (hasil == 0)
        & (asli >= 1)
        & (asli <= 5)
    )

    jumlah_kosong = int(kosong_koreksi.sum())
    hasil[kosong_koreksi] = asli[kosong_koreksi]

    # Pulihkan seluruh Farm Land yang semula
    # diprediksi RF sebagai kelas 2.
    farm_asli = asli == 2

    farm_terubah = farm_asli & (hasil != 2)
    jumlah_farm = int(farm_terubah.sum())

    hasil[farm_asli] = 2

    # Pertahankan hanya Mangrove asli RF.
    mangrove_asli = asli == 5

    mangrove_tambahan = (
        (hasil == 5)
        & (~mangrove_asli)
    )

    jumlah_mangrove_tambahan = int(mangrove_tambahan.sum())

    hasil[mangrove_tambahan] = asli[mangrove_tambahan]
    hasil[mangrove_asli] = 5

    # NoData asli tetap NoData.
    hasil[asli == 0] = 0

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

    with rasterio.open(OUTPUT_100M, "w", **profile) as dst:

        dst.write(hasil, 1)

        try:
            dst.write_colormap(1, src.colormap(1))
        except ValueError:
            pass

        dst.set_band_description(
            1,
            "RF19 - Pemulihan Farm Land dan Mangrove asli"
        )

    print("NoData hasil koreksi dipulihkan:", jumlah_kosong)
    print("Farm Land asli dipulihkan:", jumlah_farm)
    print("Mangrove tambahan dibatalkan:", jumlah_mangrove_tambahan)
    print("Jumlah Farm Land:", int((hasil == 2).sum()))
    print("Jumlah Mangrove:", int((hasil == 5).sum()))
    print("Jumlah NoData:", int((hasil == 0).sum()))

del asli, hasil


# ============================================================
# 5. RESAMPLING 100 M MENJADI GRID 50 M
# ============================================================

print("\n[2/3] Membuat raster tampilan 50 meter...")

with rasterio.open(OUTPUT_100M) as src:

    if (
        abs(src.res[0] - 100) > 0.01
        or abs(src.res[1] - 100) > 0.01
    ):
        raise ValueError("Raster input bukan resolusi 100 meter.")

    transform50 = src.transform * Affine.scale(0.5, 0.5)

    profile50 = src.profile.copy()

    profile50.update(
        width=src.width * 2,
        height=src.height * 2,
        transform=transform50,
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
        BIGTIFF="IF_SAFER"
    )

    with rasterio.open(OUTPUT_50M, "w", **profile50) as dst:

        reproject(
            source=rasterio.band(src, 1),
            destination=rasterio.band(dst, 1),
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=transform50,
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
            "RF19 - Farm Land dipulihkan - Grid tampilan 50m"
        )

print("Raster 50 meter selesai:", OUTPUT_50M.name)


# ============================================================
# 6. REPROYEKSI RASTER UNTUK PETA LEAFLET
# ============================================================

print("\n[3/3] Membuat peta interaktif...")

with rasterio.open(OUTPUT_50M) as src:

    transform, width, height = calculate_default_transform(
        src.crs,
        "EPSG:4326",
        src.width,
        src.height,
        *src.bounds
    )

    max_width = 2400
    ratio = min(1.0, max_width / width)

    new_width = max(1, round(width * ratio))
    new_height = max(1, round(height * ratio))

    transform = transform * Affine.scale(
        width / new_width,
        height / new_height
    )

    kelas = np.zeros(
        (new_height, new_width),
        dtype=np.uint8
    )

    reproject(
        source=rasterio.band(src, 1),
        destination=kelas,
        src_transform=src.transform,
        src_crs=src.crs,
        src_nodata=0,
        dst_transform=transform,
        dst_crs="EPSG:4326",
        dst_nodata=0,
        resampling=Resampling.nearest
    )

west = transform.c
north = transform.f
east = west + new_width * transform.a
south = north + new_height * transform.e

bounds = [[south, west], [north, east]]


# ============================================================
# 7. MEMBUAT PETA
# ============================================================

peta = folium.Map(
    location=[-7.65, 112.9],
    zoom_start=8,
    tiles=None,
    control_scale=True,
    prefer_canvas=True
)

folium.TileLayer(
    tiles=(
        "https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/tile/{z}/{y}/{x}"
    ),
    attr="Tiles © Esri",
    name="Citra satelit Esri",
    overlay=False,
    control=True,
    show=True
).add_to(peta)


# ============================================================
# 8. LAYER KLASIFIKASI
# ============================================================

for kode, (nama, warna, label_kelas) in KELAS.items():

    rgba = np.zeros(
        (new_height, new_width, 4),
        dtype=np.uint8
    )

    mask = kelas == kode

    rgb = tuple(
        bytes.fromhex(warna.lstrip("#"))
    )

    rgba[mask, :3] = rgb
    rgba[mask, 3] = 190

    folium.raster_layers.ImageOverlay(
        image=data_url(rgba),
        bounds=bounds,
        name=f"Klasifikasi: {nama}",
        overlay=True,
        show=True,
        opacity=1,
        interactive=False,
        control=True
    ).add_to(peta)

    print(f"Layer {nama} selesai.")


# ============================================================
# 9. SAMPEL GEOJSON
# ============================================================

files = sorted(GEOJSON_DIR.glob("*.geojson"))

if len(files) == 1:

    with files[0].open("r", encoding="utf-8") as f:
        geojson = json.load(f)

    features = geojson.get("features", [])

    print("Jumlah sampel GeoJSON:", len(features))

    for kode, (nama, warna, label_kelas) in KELAS.items():

        filtered = []

        for idx, feature in enumerate(features, 1):

            properti = feature.get("properties") or {}

            kelas_sampel = str(
                properti.get("class", "")
            ).strip().lower()

            if kelas_sampel == label_kelas:

                filtered.append({
                    "type": "Feature",
                    "geometry": feature.get("geometry"),
                    "properties": {
                        "class": label_kelas,
                        "id_sampel": idx
                    }
                })

        if not filtered:
            continue

        group = folium.FeatureGroup(
            name=f"Sampel: {nama}",
            show=False,
            overlay=True
        )

        folium.GeoJson(
            {
                "type": "FeatureCollection",
                "features": filtered
            },
            style_function=lambda ft, c=warna: {
                "color": c,
                "weight": 2,
                "fillColor": c,
                "fillOpacity": 0.14
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["class", "id_sampel"],
                aliases=["Kelas:", "ID poligon:"]
            )
        ).add_to(group)

        group.add_to(peta)

        print(f"Sampel {nama}: {len(filtered)} poligon.")

else:
    print(
        "Peringatan: folder GeoJSON harus berisi "
        "tepat satu file .geojson."
    )


# ============================================================
# 10. LEGENDA
# ============================================================

legends = "".join(
    '<div style="margin:5px 0">'
    f'<span style="background:{warna};'
    'display:inline-block;width:13px;height:13px;'
    'margin-right:8px;border-radius:3px"></span>'
    f'{nama}</div>'
    for _, (nama, warna, _) in KELAS.items()
)

legend_html = f"""
<div style="
position:fixed;
bottom:25px;
left:16px;
z-index:9999;
background:white;
padding:14px;
border-radius:10px;
box-shadow:0 4px 18px #0004;
font:13px Arial;
color:#222;
">
<strong>Legenda Tutupan Lahan</strong>
{legends}
<hr>
RF 19 fitur - pemulihan sawah
<div style="font-size:11px;color:#666;margin-top:5px">
Grid tampilan 50 m; data asli 100 m
</div>
</div>
"""

peta.get_root().html.add_child(
    folium.Element(legend_html)
)

folium.LayerControl(
    collapsed=False,
    position="topright"
).add_to(peta)

peta.fit_bounds(bounds)

peta.save(str(OUTPUT_HTML))


# ============================================================
# 11. HASIL
# ============================================================

print("\n" + "=" * 65)
print("SEMUA PROSES SELESAI")
print("=" * 65)

print("Raster 100m:", OUTPUT_100M)
print("Raster 50m :", OUTPUT_50M)
print("Peta HTML  :", OUTPUT_HTML)

print("\nModel RF tetap 19 fitur.")
print("Model dan raster asli tidak diubah.")
print("Farm Land dan Mangrove mengikuti prediksi RF asli.")
print("Koreksi Water Body lainnya masih bersifat eksperimental.")
