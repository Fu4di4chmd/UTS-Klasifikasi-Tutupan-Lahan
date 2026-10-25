"""Peta perbandingan kelas Mangrove: model 14 fitur vs revisi 17 fitur.
Jalankan dari root proyek: py -3.9 scripts/15_peta_perbandingan_mangrove.py
Dependensi: numpy, rasterio, folium, pillow, geopandas (opsional untuk sampel).
Tidak mengubah raster/model/website lama.
"""
from pathlib import Path
import base64
from io import BytesIO
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, transform_bounds
from PIL import Image
import folium
from folium.plugins import Fullscreen, MousePosition

BASE = Path(__file__).resolve().parent.parent
MAPS = BASE / 'outputs' / 'maps'
OLD = MAPS / 'klasifikasi_rf_jatim_sampling_poligon.tif'
NEW = MAPS / 'klasifikasi_rf_jatim_mangrove_17fitur.tif'
OUT = MAPS / 'perbandingan_mangrove_14_vs_17.html'
# Ukuran batas tampilan agar HTML cukup ringan; klasifikasi agregat nearest-neighbor.
MAX_SIDE = 2200

for p in (OLD, NEW):
    if not p.exists():
        raise FileNotFoundError(f'Raster tidak ditemukan: {p}')

with rasterio.open(OLD) as old, rasterio.open(NEW) as new:
    if (old.width, old.height, old.crs, old.transform) != (new.width, new.height, new.crs, new.transform):
        raise ValueError('Raster lama dan baru harus memiliki grid/CRS/transform yang sama.')
    factor = max(old.width / MAX_SIDE, old.height / MAX_SIDE, 1)
    width, height = int(np.ceil(old.width / factor)), int(np.ceil(old.height / factor))
    old_sm = old.read(1, out_shape=(height, width), resampling=Resampling.nearest)
    new_sm = new.read(1, out_shape=(height, width), resampling=Resampling.nearest)
    # Derive scaled transform; reprojection into EPSG:4326 for correct Leaflet geolocation.
    src_transform = old.transform * old.transform.scale(old.width / width, old.height / height)
    bounds = transform_bounds(old.crs, 'EPSG:4326', *old.bounds, densify_pts=41)

# Reproject to true geographic equirectangular grid, preserves raster map alignment.
west, south, east, north = bounds
lat_mid = (south + north) / 2
# choose geographic shape accounting for latitude aspect
out_width = width
out_height = max(1, int(round(height)))
from rasterio.transform import from_bounds
geo_transform = from_bounds(west, south, east, north, out_width, out_height)

def to_geographic(a):
    b = np.zeros((out_height, out_width), dtype=np.uint8)
    reproject(source=a, destination=b, src_transform=src_transform,
              src_crs='EPSG:32749' if False else raster_crs,
              dst_transform=geo_transform, dst_crs='EPSG:4326',
              src_nodata=0, dst_nodata=0, resampling=Resampling.nearest)
    return b

with rasterio.open(OLD) as src:
    raster_crs = src.crs

old_geo = to_geographic(old_sm)
new_geo = to_geographic(new_sm)
a = old_geo == 5
b = new_geo == 5
common = a & b
added = ~a & b
removed = a & ~b


def encoded_mask(mask, rgb):
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[mask, :3] = rgb
    rgba[mask, 3] = 210
    bio = BytesIO()
    Image.fromarray(rgba, 'RGBA').save(bio, format='PNG', optimize=True)
    return 'data:image/png;base64,' + base64.b64encode(bio.getvalue()).decode('ascii')

m = folium.Map(location=[lat_mid, (west+east)/2], zoom_start=8, tiles=None, control_scale=True, prefer_canvas=True)
folium.TileLayer('OpenStreetMap', name='OpenStreetMap', overlay=False).add_to(m)
folium.TileLayer(tiles='Esri.WorldImagery', attr='Esri', name='Citra Satelit Esri', overlay=False).add_to(m)
geo_bounds = [[south, west], [north, east]]

items = [
    ('Mangrove model lama (14 fitur)', a, (255, 165, 0), False),
    ('Mangrove model revisi (17 fitur, 305 poligon)', b, (0, 210, 100), True),
    ('Mangrove tetap (kedua model)', common, (40, 160, 235), False),
    ('Mangrove baru ditambahkan', added, (239, 50, 50), False),
    ('Mangrove hilang dari model revisi', removed, (255, 224, 34), False),
]
for name, mask, rgb, show in items:
    folium.raster_layers.ImageOverlay(
        image=encoded_mask(mask, rgb), bounds=geo_bounds, name=name,
        overlay=True, show=show, opacity=0.8, interactive=False,
        cross_origin=False, zindex=5,
    ).add_to(m)

# Sample polygons help diagnose training geometry versus imagery.
geojson_dir = BASE / 'data' / 'geojson'
gfiles = list(geojson_dir.glob('*.geojson'))
if len(gfiles) == 1:
    try:
        import geopandas as gpd
        sample = gpd.read_file(gfiles[0])
        sample['class'] = sample['class'].astype(str).str.strip().str.lower()
        sample = sample.loc[sample['class'] == 'mangrove'].to_crs('EPSG:4326').copy()
        sample['sample_id'] = sample.index.astype(int) + 1
        folium.GeoJson(
            sample[['sample_id', 'class', 'geometry']].to_json(),
            name=f'Sampel GeoJSON Mangrove ({len(sample)} poligon)', show=False,
            style_function=lambda _: {'color':'#ff33dd','weight':2,'fillOpacity':0.03},
            tooltip=folium.GeoJsonTooltip(fields=['sample_id', 'class'], aliases=['ID baris:', 'Kelas:'])
        ).add_to(m)
        print('Poligon Mangrove ditambahkan:', len(sample))
    except Exception as e:
        print('Poligon sampel tidak ditampilkan:', e)

folium.LayerControl(collapsed=False).add_to(m)
Fullscreen(position='topleft').add_to(m)
MousePosition(position='bottomleft').add_to(m)
legend = '''<div style="position:fixed;bottom:25px;right:25px;z-index:9999;background:white;
padding:12px 16px;border:1px solid #aaa;border-radius:8px;font:13px Arial;max-width:270px">
<b>Perbandingan Mangrove</b><br>
<span style="color:#ffa500">■</span> Lama (14 fitur)<br>
<span style="color:#00d264">■</span> Revisi (17 fitur, 305 poligon)<br>
<span style="color:#28a0eb">■</span> Tetap Mangrove<br>
<span style="color:#ef3232">■</span> Bertambah<br>
<span style="color:#ffe022">■</span> Hilang<br>
<span style="color:#ff33dd">▢</span> Sampel GeoJSON<br>
<small>Aktifkan layer satu per satu untuk membaca perubahan. Data overlay diperkecil untuk navigasi; gunakan TIFF asli untuk hitung statistik.</small></div>'''
m.get_root().html.add_child(folium.Element(legend))
MAPS.mkdir(parents=True, exist_ok=True)
m.save(str(OUT))
print('Peta tersimpan:', OUT)
print('Buka dengan browser atau: start "" "' + str(OUT) + '"')
print('Catatan: layer perubahan bersifat visual pada resolusi tampilan; bukan ukuran piksel asli.')
