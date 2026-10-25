"""Audit poligon Mangrove terhadap grid dan nilai Sentinel-2 (tanpa mengubah data).
Jalankan dari folder proyek: py -3.9 scripts/08_cek_sampel_mangrove_baru.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window

BASE = Path(__file__).resolve().parent.parent
GEOJSON_DIR = BASE / 'data' / 'geojson'
RASTER_PATH = BASE / 'data' / 'prediksi' / 'sentinel2_jatim_training_buffer3km_14fitur_100m.tif'
OUTPUT_DIR = BASE / 'outputs' / 'reports' / 'audit_mangrove_baru'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BANDS = ['B02','B03','B04','B05','B06','B07','B08','B8A','B11','B12',
         'NDVI','NDMI','MNDWI','NDBI']

files = sorted(GEOJSON_DIR.glob('*.geojson'))
if len(files) != 1:
    raise RuntimeError(f'Harus tepat satu GeoJSON dalam {GEOJSON_DIR}; ditemukan {len(files)}: {[p.name for p in files]}')
if not RASTER_PATH.is_file():
    raise FileNotFoundError(RASTER_PATH)

gdf = gpd.read_file(files[0])
if gdf.crs is None or 'class' not in gdf.columns:
    raise ValueError('GeoJSON harus memiliki CRS dan kolom class.')
gdf['class'] = gdf['class'].astype(str).str.strip().str.lower()
gdf['polygon_id'] = np.arange(1, len(gdf) + 1)
mang = gdf[gdf['class'] == 'mangrove'].copy()
if mang.empty:
    raise ValueError('Tidak ada poligon berlabel mangrove.')

print('GeoJSON:', files[0].name, '| total poligon:', len(gdf), '| mangrove:', len(mang))
with rasterio.open(RASTER_PATH) as src:
    if src.count != 14 or list(src.descriptions) != BANDS:
        raise ValueError('Raster training bukan raster 14 fitur dengan urutan band yang diharapkan.')
    gdf = gdf.to_crs(src.crs)
    mang = mang.to_crs(src.crs)
    # Meniru preprocessing: rasterisasi semua poligon sekaligus. Poligon yang
    # saling bertumpukan (jika ada) mengikuti urutan terakhir.
    ids = rasterize(
        ((geom, int(pid)) for geom, pid in zip(gdf.geometry, gdf.polygon_id)),
        out_shape=(src.height, src.width), transform=src.transform,
        fill=0, dtype='int32',
    )
    # Pixel mask secara bertahap agar tidak menampung 14 band sekaligus.
    valid = ids > 0
    finite = ids > 0
    for i, name in enumerate(BANDS, start=1):
        band = src.read(i, masked=True)
        a = np.asarray(band.filled(np.nan), dtype='float32')
        ok = np.isfinite(a)
        finite &= ok
        valid &= ok
        if i <= 10:
            valid &= a > 0
        print(f'Pemeriksaan band {i:02d}/14: {name}')
        del band, a

    rows = []
    for r in mang.itertuples():
        pid = int(r.polygon_id)
        # Seluruh piksel di dalam poligon menurut rasterisasi sendiri (untuk
        # membedakan masalah overlap dari piksel yang terlalu kecil).
        single = rasterize([(r.geometry, 1)], out_shape=(src.height, src.width),
                           transform=src.transform, fill=0, dtype='uint8')
        own_pixels = int(np.count_nonzero(single))
        assigned = ids == pid
        raster_pixels = int(np.count_nonzero(assigned))
        good = int(np.count_nonzero(valid & assigned))
        finite_count = int(np.count_nonzero(finite & assigned))
        if own_pixels == 0:
            reason = 'Tidak mengenai pusat piksel raster 100 m (atau di luar raster)'
        elif raster_pixels == 0:
            reason = 'Piksel tertimpa poligon lain saat rasterisasi'
        elif good == 0:
            reason = 'Ada piksel, tetapi tidak lolos filter band'
        else:
            reason = 'Ada piksel valid'
        center = r.geometry.representative_point()
        rows.append({
            'polygon_id': pid, 'class': 'mangrove',
            'lon': float(gpd.GeoSeries([center], crs=src.crs).to_crs('EPSG:4326').iloc[0].x),
            'lat': float(gpd.GeoSeries([center], crs=src.crs).to_crs('EPSG:4326').iloc[0].y),
            'area_m2': round(float(r.geometry.area), 1),
            'pixels_if_alone': own_pixels,
            'pixels_assigned': raster_pixels,
            'pixels_finite_14': finite_count,
            'pixels_valid_14': good,
            'status': reason,
        })

report = pd.DataFrame(rows)
path = OUTPUT_DIR / 'diagnosis_80_poligon_mangrove.csv'
report.to_csv(path, index=False, encoding='utf-8-sig')
print('\n' + '=' * 60)
print('AUDIT POLIGON MANGROVE SELESAI')
print('=' * 60)
print('Total Mangrove:', len(report))
print('Poligon dengan piksel valid:', int((report.pixels_valid_14 > 0).sum()))
print('Poligon tanpa piksel valid:', int((report.pixels_valid_14 == 0).sum()))
print('Total piksel valid Mangrove:', int(report.pixels_valid_14.sum()))
print('\nPenyebab/status:\n' + report.status.value_counts().to_string())
print('\n10 poligon dengan piksel valid paling sedikit:')
print(report.sort_values(['pixels_valid_14','polygon_id']).head(10)[
    ['polygon_id','pixels_if_alone','pixels_assigned','pixels_valid_14','status']
].to_string(index=False))
print('\nCSV:', path)
print('GeoJSON asli tidak diubah.')
