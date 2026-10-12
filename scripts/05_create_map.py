
# ==========================================================
# TAHAP 5 - PETA INTERAKTIF LEAFLET
# SENTINEL-2 TRUE COLOR + GEOJSON
# ==========================================================

from pathlib import Path
import numpy as np
import geopandas as gpd
import rasterio
import folium

from PIL import Image
from rasterio.warp import (
    calculate_default_transform,
    reproject,
    Resampling
)
from rasterio.transform import Affine
from branca.element import Element

# ==========================================================
# 1. LOKASI FOLDER
# ==========================================================

BASE = Path(__file__).resolve().parent.parent

GEOJSON_DIR = BASE / "data" / "geojson"
RASTER_DIR = BASE / "data" / "sentinel2"
MAP_DIR = BASE / "outputs" / "maps"

MAP_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# 2. BACA GEOJSON
# ==========================================================

geojson_files = list(GEOJSON_DIR.glob("*.geojson"))

if len(geojson_files) != 1:
    raise ValueError(
        "Folder data/geojson harus berisi satu GeoJSON."
    )

gdf = gpd.read_file(geojson_files[0])

if gdf.crs is None:
    raise ValueError("CRS GeoJSON tidak ditemukan.")

gdf["class"] = (
    gdf["class"]
    .astype(str)
    .str.strip()
    .str.lower()
)

gdf = gdf.to_crs("EPSG:4326")

print("Jumlah poligon:", len(gdf))
print(gdf["class"].value_counts())

# ==========================================================
# 3. CARI TIFF TRUE COLOR
# ==========================================================

tiff_files = list(RASTER_DIR.rglob("*.tif"))
tiff_files += list(RASTER_DIR.rglob("*.tiff"))

truecolor_files = [
    p for p in tiff_files
    if "true_color" in p.name.lower()
]

if len(truecolor_files) != 1:
    raise ValueError(
        "Harus ada tepat satu TIFF True Color."
    )

truecolor_path = truecolor_files[0]

print("\nTrue Color:", truecolor_path.name)

# ==========================================================
# 4. REPROYEKSI TRUE COLOR KE EPSG:4326
# ==========================================================

with rasterio.open(truecolor_path) as src:

    if src.count < 3:
        raise ValueError("True Color harus memiliki 3 band.")

    dst_transform, dst_width, dst_height = (
        calculate_default_transform(
            src.crs,
            "EPSG:4326",
            src.width,
            src.height,
            *src.bounds
        )
    )

    # Perkecil gambar agar ringan dibuka di browser
    max_dimension = 1600
    scale = max(dst_width, dst_height) / max_dimension

    if scale > 1:
        dst_width = max(1, int(np.ceil(dst_width / scale)))
        dst_height = max(1, int(np.ceil(dst_height / scale)))
        dst_transform = (
            dst_transform * Affine.scale(scale, scale)
        )

    rgb = np.zeros(
        (3, dst_height, dst_width),
        dtype=np.float32
    )

    mask = np.zeros(
        (dst_height, dst_width),
        dtype=np.uint8
    )

    # Mask untuk area yang benar-benar memiliki citra
    reproject(
        source=src.dataset_mask(),
        destination=mask,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=dst_transform,
        dst_crs="EPSG:4326",
        src_nodata=0,
        dst_nodata=0,
        resampling=Resampling.nearest
    )

    for i in range(3):
        reproject(
            source=rasterio.band(src, i + 1),
            destination=rgb[i],
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs="EPSG:4326",
            resampling=Resampling.bilinear
        )

# ==========================================================
# 5. KONVERSI RGB MENJADI PNG TRANSPARAN
# ==========================================================

rgb = np.nan_to_num(rgb)

if np.max(rgb) <= 1.5:
    rgb *= 255.0

rgb_uint8 = np.moveaxis(
    np.clip(rgb, 0, 255).astype(np.uint8),
    0,
    -1
)

alpha = np.where(mask > 0, 255, 0).astype(np.uint8)

rgba = np.dstack([rgb_uint8, alpha])

image_path = MAP_DIR / "sentinel2_truecolor.png"

Image.fromarray(rgba, mode="RGBA").save(image_path)

print("✅ Gambar True Color dibuat:", image_path.name)

# ==========================================================
# 6. BATAS PETA
# ==========================================================

west, north = dst_transform * (0, 0)
east, south = dst_transform * (dst_width, dst_height)

bounds = [
    [south, west],
    [north, east]
]

# ==========================================================
# 7. BUAT PETA LEAFLET
# ==========================================================

m = folium.Map(
    location=[
        (south + north) / 2,
        (west + east) / 2
    ],
    zoom_start=9,
    tiles=None,
    control_scale=True
)

# Gunakan citra milik sendiri sebagai latar
folium.raster_layers.ImageOverlay(
    image=str(image_path),
    bounds=bounds,
    name="Sentinel-2 True Color",
    overlay=False,
    control=False,
    opacity=1
).add_to(m)

# ==========================================================
# 8. LAYER GEOJSON PER KELAS
# ==========================================================

kelas_warna = {
    "build up": ("Built-up", "#808080"),
    "farm land": ("Farmland", "#FFD700"),
    "water body": ("Water Body", "#2196F3"),
    "forest": ("Forest", "#228B22"),
    "mangrove": ("Mangrove", "#00CED1")
}

for kode, (nama, warna) in kelas_warna.items():

    bagian = gdf[gdf["class"] == kode].copy()

    grup = folium.FeatureGroup(
        name=nama,
        show=True
    )

    folium.GeoJson(
        bagian.to_json(),
        style_function=lambda feature, c=warna: {
            "color": c,
            "weight": 2,
            "fillColor": c,
            "fillOpacity": 0.55
        },
        highlight_function=lambda feature: {
            "color": "#FFFFFF",
            "weight": 3,
            "fillOpacity": 0.75
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["class"],
            aliases=["Kelas:"]
        )
    ).add_to(grup)

    grup.add_to(m)

    print(f"✅ {nama}: {len(bagian)} poligon")

# Checkbox layer
folium.LayerControl(
    collapsed=False,
    position="topright"
).add_to(m)

# ==========================================================
# 9. JUDUL PETA
# ==========================================================

title_html = """
<div style="
position:fixed;
top:10px;
left:55px;
z-index:9999;
background:white;
padding:12px 16px;
border-radius:8px;
font-family:Arial;
box-shadow:0 2px 8px #0002;
">
<b>PETA TUTUPAN LAHAN JAWA TIMUR</b>
<div style="font-size:12px;margin-top:5px;">
Sentinel-2 True Color | GeoJSON Digitasi
</div>
</div>
"""

m.get_root().html.add_child(Element(title_html))

# ==========================================================
# 10. LEGENDA
# ==========================================================

legend_html = """
<div style="
position:fixed;
bottom:30px;
right:15px;
z-index:9999;
background:white;
padding:12px 16px;
border-radius:8px;
font-family:Arial;
font-size:13px;
box-shadow:0 2px 8px #0002;
">
<b>LEGENDA TUTUPAN LAHAN</b>
<hr>
<div><span style="background:#808080;padding:0 9px;margin-right:8px;"></span>Built-up</div>
<div><span style="background:#FFD700;padding:0 9px;margin-right:8px;"></span>Farmland</div>
<div><span style="background:#2196F3;padding:0 9px;margin-right:8px;"></span>Water Body</div>
<div><span style="background:#228B22;padding:0 9px;margin-right:8px;"></span>Forest</div>
<div><span style="background:#00CED1;padding:0 9px;margin-right:8px;"></span>Mangrove</div>
</div>
"""

m.get_root().html.add_child(Element(legend_html))

# ==========================================================
# 11. FOKUS PETA KE LOKASI DIGITASI
# ==========================================================

xmin, ymin, xmax, ymax = gdf.total_bounds

m.fit_bounds([
    [ymin, xmin],
    [ymax, xmax]
])

# ==========================================================
# 12. SIMPAN HTML
# ==========================================================

output_html = MAP_DIR / "peta_interaktif.html"

m.save(str(output_html))

print("\n" + "=" * 55)
print("✅ PETA INTERAKTIF BERHASIL DIBUAT")
print("=" * 55)

print("Total poligon:", len(gdf))
print("Total kelas:", len(kelas_warna))
print("File:", output_html)
print("✅ Peta siap digunakan pada website.")
