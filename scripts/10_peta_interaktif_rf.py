"""Peta Leaflet hasil Random Forest dan sampel GeoJSON untuk Streamlit.
Jalankan dari root: py -3.9 scripts/10_peta_interaktif_rf.py
Tidak mengubah klasifikasi raster atau model.
"""
from pathlib import Path
from io import BytesIO
import base64
import json

import folium
import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject

BASE = Path(__file__).resolve().parent.parent
RASTER_RF = BASE / 'outputs/maps/klasifikasi_rf_jatim_sampling_poligon.tif'
RASTER_S2 = BASE / 'data/prediksi/sentinel2_jatim_14fitur_100m.tif'
GEOJSON_DIR = BASE / 'data/geojson'
OUTPUT = BASE / 'outputs/maps/peta_interaktif_rf_sampling_poligon.html'

KELAS = {
    1: ('Built-up', '#d45b56', 'build up'),
    2: ('Farm land', '#edaa63', 'farm land'),
    3: ('Water body', '#298fe8', 'water body'),
    4: ('Forest', '#208653', 'forest'),
    5: ('Mangrove', '#04bfb4', 'mangrove'),
}


def to_data_url(array):
    buffer = BytesIO()
    Image.fromarray(array).save(buffer, format='PNG', optimize=True)
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')


def reprojection_grid(src, max_width=2400):
    transform, width, height = calculate_default_transform(
        src.crs, 'EPSG:4326', src.width, src.height, *src.bounds
    )
    ratio = min(1.0, max_width / width)
    new_width = max(1, int(round(width * ratio)))
    new_height = max(1, int(round(height * ratio)))
    if ratio != 1:
        transform = transform * rasterio.Affine.scale(width / new_width, height / new_height)
    west, north = transform.c, transform.f
    east = west + new_width * transform.a
    south = north + new_height * transform.e
    return transform, new_width, new_height, [[south, west], [north, east]]


def warper(src, band, transform, width, height, nodata):
    out = np.zeros((height, width), dtype=np.float32)
    reproject(
        source=rasterio.band(src, band), destination=out,
        src_transform=src.transform, src_crs=src.crs,
        src_nodata=src.nodata if src.nodata is not None else nodata,
        dst_transform=transform, dst_crs='EPSG:4326',
        dst_nodata=nodata, resampling=Resampling.nearest,
    )
    return out


def make_satellite_true_color(map_obj):
    if not RASTER_S2.exists():
        print('Citra Sentinel-2 lokal tidak tersedia; opsi Sentinel-2 dilewati.')
        return
    with rasterio.open(RASTER_S2) as src:
        if src.count < 4:
            print('Band Sentinel-2 tidak lengkap; opsi Sentinel-2 dilewati.')
            return
        transform, width, height, bounds = reprojection_grid(src, max_width=1700)
        bands = [warper(src, i, transform, width, height, 0) for i in (3, 2, 1)]
        rgb = np.stack(bands, axis=-1)
        valid = np.isfinite(rgb).all(axis=2) & (rgb > 0).all(axis=2)
        out = np.zeros((height, width, 4), np.uint8)
        if valid.any():
            for j in range(3):
                channel = rgb[:, :, j]
                p_lo, p_hi = np.percentile(channel[valid], [2, 98])
                # Nilai NoData/NaN tidak ikut dikonversi ke uint8.
                stretched = np.clip(
                    (channel[valid] - p_lo) / max(p_hi - p_lo, 1e-6), 0, 1
                )
                out[:, :, j][valid] = np.uint8(np.round(stretched ** 0.9 * 255))
            out[:, :, 3][valid] = 255
        folium.raster_layers.ImageOverlay(
            image=to_data_url(out), bounds=bounds,
            name='Citra Sentinel-2 (B04/B03/B02)',
            overlay=False, show=False, interactive=False,
        ).add_to(map_obj)
        print('Basemap Sentinel-2 lokal selesai.')


if not RASTER_RF.exists():
    raise FileNotFoundError(f'Raster prediksi RF tidak ditemukan: {RASTER_RF}')

print('Membaca hasil Random Forest...')
with rasterio.open(RASTER_RF) as src:
    transform, width, height, bounds = reprojection_grid(src, max_width=2400)
    kelas_raster = np.zeros((height, width), np.uint8)
    reproject(
        source=rasterio.band(src, 1), destination=kelas_raster,
        src_transform=src.transform, src_crs=src.crs,
        src_nodata=0, dst_transform=transform, dst_crs='EPSG:4326',
        dst_nodata=0, resampling=Resampling.nearest,
    )

peta = folium.Map(
    location=[-7.65, 112.9], tiles=None, zoom_start=8,
    control_scale=True, prefer_canvas=True,
)

# Gunakan basemap Esri. OpenStreetMap dihilangkan karena tile 403 Access blocked.
folium.TileLayer(
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attr='Tiles © Esri', name='Citra satelit global (Esri)',
    overlay=False, control=True, show=True,
).add_to(peta)
make_satellite_true_color(peta)

for kode, (nama, warna, label) in KELAS.items():
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    mask = kelas_raster == kode
    color = tuple(bytes.fromhex(warna.lstrip('#')))
    rgba[mask, :3] = color
    rgba[mask, 3] = 190
    folium.raster_layers.ImageOverlay(
        image=to_data_url(rgba), bounds=bounds,
        name=f'Klasifikasi: {nama}', show=False, opacity=1.0,
        interactive=False, overlay=True, control=True,
    ).add_to(peta)
    print(f'Layer klasifikasi {nama} selesai.')

geojson_files = list(GEOJSON_DIR.glob('*.geojson'))
if len(geojson_files) == 1:
    with geojson_files[0].open('r', encoding='utf-8') as f:
        geojson = json.load(f)
    features = geojson.get('features', [])
    for kode, (nama, warna, label) in KELAS.items():
        filtered = []
        for idx, ft in enumerate(features, 1):
            if str((ft.get('properties') or {}).get('class', '')).strip().lower() == label:
                # Properti tanpa spasi agar style_function Folium membuat JS valid.
                filtered.append({
                    'type': 'Feature',
                    'geometry': ft.get('geometry'),
                    'properties': {'class': label, 'id_sampel': idx},
                })
        if not filtered:
            continue
        group = folium.FeatureGroup(name=f'Sampel: {nama}', show=False, overlay=True)
        folium.GeoJson(
            {'type': 'FeatureCollection', 'features': filtered},
            style_function=lambda feature, c=warna: {
                'color': c, 'weight': 2.5, 'opacity': 1,
                'fillColor': c, 'fillOpacity': 0.14,
            },
            highlight_function=lambda feature: {'weight': 4, 'fillOpacity': 0.28},
            tooltip=folium.GeoJsonTooltip(
                fields=['class', 'id_sampel'], aliases=['Kelas:', 'ID poligon:'],
            ),
        ).add_to(group)
        group.add_to(peta)
        print(f'Layer sampel {nama}: {len(filtered)} poligon.')
else:
    print('PERHATIAN: Folder data/geojson harus berisi tepat satu .geojson untuk layer sampel.')

legends = ''.join(
    f'<div style="display:flex;align-items:center;gap:8px;margin:5px 0">'
    f'<span style="background:{color};width:13px;height:13px;display:inline-block;border-radius:3px"></span>'
    f'<span>{name}</span></div>'
    for _, (name, color, _) in KELAS.items()
)
legend_html = f'''<div style="position:fixed;bottom:26px;left:16px;z-index:9999;
background:rgba(255,255,255,.96);color:#18202c;padding:13px 16px;
box-shadow:0 4px 22px rgba(0,0,0,.22);border-radius:12px;
font:13px Arial,sans-serif;min-width:155px">
<strong style="font-size:14px">Legenda Tutupan Lahan</strong>{legends}</div>'''
peta.get_root().html.add_child(folium.Element(legend_html))
folium.LayerControl(collapsed=False, position='topright', autoZIndex=False).add_to(peta)
peta.fit_bounds(bounds)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
peta.save(str(OUTPUT))
print('\nPETA INTERAKTIF SELESAI')
print('Lokasi:', OUTPUT)
print('Catatan: Layer klasifikasi adalah prediksi asli Random Forest (belum dikoreksi).')
