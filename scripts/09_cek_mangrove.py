"""Diagnosis spasial prediksi Mangrove RF 14 vs 17 fitur.

Jalankan dari root proyek: py -3.9 scripts/09_cek_mangrove.py
Membaca peta dan DEM yang sudah ada; TIDAK mengubah prediksi/model.
Natural Earth 1:10m adalah garis pantai tergeneralisasi, hanya untuk diagnosis.
"""
from pathlib import Path
import csv

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_bounds
from scipy.ndimage import distance_transform_edt

BASE = Path(__file__).resolve().parent.parent
MAPS = BASE / 'outputs' / 'maps'
OLD = MAPS / 'klasifikasi_rf_jatim_sampling_poligon.tif'
NEW = MAPS / 'klasifikasi_rf_jatim_mangrove_17fitur.tif'
DEM = BASE / 'data' / 'prediksi' / 'dem_jatim_aligned_100m.tif'
COAST = BASE / 'data' / 'coastline' / 'ne_10m_coastline.shp'
REPORT = BASE / 'outputs' / 'reports' / 'eksperimen_mangrove_17fitur' / 'diagnosis_jarak_pesisir_14_vs_17.csv'

for p in (OLD, NEW, DEM, COAST):
    if not p.is_file():
        raise FileNotFoundError(f'File tidak ditemukan: {p}')

print('=' * 65)
print('DIAGNOSIS JARAK KE PANTAI - MANGROVE 14 VS 17 FITUR')
print('=' * 65)

with rasterio.open(OLD) as old_src, rasterio.open(NEW) as new_src, rasterio.open(DEM) as dem_src:
    for other, name in [(new_src, 'peta baru'), (dem_src, 'DEM')]:
        if (other.width != old_src.width or other.height != old_src.height
                or other.transform != old_src.transform or other.crs != old_src.crs):
            raise ValueError(f'Grid {name} tidak sama dengan peta lama; perlu alignment lebih dulu.')

    if not old_src.crs.is_projected:
        raise ValueError('Raster harus dalam proyeksi meter untuk menghitung jarak.')

    # Baca hanya fitur pantai di sekitar cakupan raster, ditambah buffer sekitar 30 km.
    x0, y0, x1, y1 = old_src.bounds
    bbox = transform_bounds(old_src.crs, 'EPSG:4326', x0-30000, y0-30000,
                            x1+30000, y1+30000, densify_pts=21)
    shore = gpd.read_file(COAST, bbox=bbox)
    if shore.empty:
        raise ValueError('Tidak ditemukan garis pantai pada cakupan raster.')
    shore = shore[shore.geometry.notna() & ~shore.geometry.is_empty].to_crs(old_src.crs)
    print('Fitur garis pantai terbaca:', len(shore))

    coast_pixels = rasterize(
        ((g, 1) for g in shore.geometry),
        out_shape=(old_src.height, old_src.width),
        transform=old_src.transform, fill=0, dtype='uint8', all_touched=True
    ).astype(bool)
    if not coast_pixels.any():
        raise ValueError('Tidak ada garis pantai yang melewati piksel raster.')

    # scipy menghitung jarak Euclidean ke piksel garis pantai terdekat.
    # Jarak dalam meter; pendekatan berbasis grid 100 m, bukan jarak habitat ekologis.
    yres, xres = abs(old_src.res[1]), abs(old_src.res[0])
    print('Menghitung jarak ke garis pantai untuk raster ...')
    dist_m = distance_transform_edt(~coast_pixels, sampling=(yres, xres))
    del coast_pixels

    old_arr = old_src.read(1, masked=True)
    new_arr = new_src.read(1, masked=True)
    dem_arr = dem_src.read(1, masked=True)

    common = (~np.ma.getmaskarray(old_arr) & ~np.ma.getmaskarray(new_arr)
              & np.isin(old_arr.data, [1,2,3,4,5])
              & np.isin(new_arr.data, [1,2,3,4,5]))
    old_m = common & (old_arr.data == 5)
    new_m = common & (new_arr.data == 5)
    added = new_m & ~old_m
    removed = old_m & ~new_m
    dem_valid = (~np.ma.getmaskarray(dem_arr)
                 & np.isfinite(dem_arr.data))

    def count(m):
        return int(np.count_nonzero(m))

    masks = {'14 fitur': old_m, '17 fitur': new_m,
             'Mangrove bertambah': added, 'Mangrove berkurang': removed}
    bins = [(0, 2000, '0–2 km'), (2000, 5000, '2–5 km'),
            (5000, 10000, '5–10 km'), (10000, np.inf, '>10 km')]
    rows = []
    for name, m in masks.items():
        total = count(m)
        print('\n' + name + f': {total:,} piksel')
        for a, b, label in bins:
            n = count(m & (dist_m >= a) & (dist_m < b))
            pct = (100 * n / total) if total else 0
            print(f'  Jarak {label:>7}: {n:>8,} ({pct:5.1f}%)')
            rows.append({'kelompok': name, 'indikator': 'jarak_pantai',
                         'kategori': label, 'piksel': n, 'total_kelompok': total,
                         'persen_dari_kelompok': round(pct, 2)})
        high = count(m & dem_valid & (dem_arr.data > 300))
        print(f'  Elevasi >300 m: {high:,} piksel (DEM valid)')
        rows.append({'kelompok': name, 'indikator': 'elevasi',
                     'kategori': '>300 m', 'piksel': high, 'total_kelompok': total,
                     'persen_dari_kelompok': round(100*high/total, 2) if total else 0})

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open('w', newline='', encoding='utf-8-sig') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('\nLaporan:', REPORT)
    print('Catatan: Natural Earth adalah garis pantai umum (bukan habitat mangrove).')
    print('Jarak pantai dan elevasi hanya indikator diagnostik, bukan bukti kebenaran.')
