"""Tahap 31: validasi, mosaik, dan clip raster klasifikasi Jawa Timur.
Jalankan dari root proyek: py -3.9 scripts/31_gabung_raster_jatim.py
Dependensi: rasterio, geopandas, numpy.
"""
from pathlib import Path
from collections import Counter
import math
import json
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window, from_bounds, transform as window_transform
from rasterio.features import geometry_mask
from rasterio.crs import CRS

BASE = Path(__file__).resolve().parent.parent
INPUT = BASE / 'outputs/maps/klasifikasi_20m'
GEOJSON = BASE / 'data/geojson/batas_jawa_timur.geojson'
OUT = BASE / 'outputs/maps/klasifikasi_jawa_timur_20m.tif'
REPORT = BASE / 'outputs/reports/31_laporan_mosaik_jatim.json'
N_EXPECTED = 195
CRS_TARGET = CRS.from_epsg(32749)
RES = 20
CLASS_NAMES = {'1':'Built-up', '2':'Farm Land', '3':'Water Body', '4':'Forest', '5':'Mangrove'}
COLORMAP = {0:(0,0,0,0),1:(230,57,70,255),2:(245,196,66,255),3:(48,117,211,255),4:(45,148,81,255),5:(155,85,182,255)}


def main():
    rasters = sorted(INPUT.glob('klasifikasi_rf_B_*_20m.tif'))
    print(f'Raster ditemukan: {len(rasters)}/{N_EXPECTED}', flush=True)
    if len(rasters) != N_EXPECTED:
        raise RuntimeError('Jumlah raster belum 195. Pastikan semua blok lengkap sebelum penggabungan.')
    if not GEOJSON.is_file():
        raise FileNotFoundError(f'Batas administrasi tidak ditemukan: {GEOJSON}')

    bounds_list = []
    for i, path in enumerate(rasters, 1):
        with rasterio.open(path) as src:
            if src.crs != CRS_TARGET or src.count != 1 or src.res != (RES, RES):
                raise ValueError(f'CRS/band/resolusi tidak sesuai: {path.name} | {src.crs} {src.count} {src.res}')
            if src.dtypes[0] not in ('uint8','int8','uint16','int16','int32','uint32'):
                raise ValueError(f'Data kelas bukan integer: {path.name} = {src.dtypes[0]}')
            # Tes pembacaan raster seluruhnya untuk menangkap kerusakan/tidak lengkap.
            histogram = np.zeros(256, dtype=np.int64)
            for _, win in src.block_windows(1):
                a = src.read(1, window=win)
                if np.any((a < 0) | (a > 5)):
                    raise ValueError(f'Nilai kelas di luar 0..5 pada {path.name}')
                histogram[:6] += np.bincount(a.ravel().astype('uint8'), minlength=6)[:6]
            if histogram[1:].sum() == 0:
                raise ValueError(f'Raster seluruhnya NoData: {path.name}')
            bounds_list.append(src.bounds)
        if i % 25 == 0 or i == len(rasters):
            print(f'[VALIDASI] {i}/{len(rasters)}', flush=True)

    gdf = gpd.read_file(GEOJSON)
    if gdf.empty or gdf.crs is None:
        raise ValueError('GeoJSON batas provinsi kosong atau tanpa CRS.')
    geom = gdf.to_crs(CRS_TARGET).geometry.unary_union
    if geom.is_empty:
        raise ValueError('Geometri batas provinsi kosong.')
    from shapely.geometry import mapping
    boundary = [mapping(geom)]

    minx = RES * math.floor(min(b.left for b in bounds_list) / RES)
    miny = RES * math.floor(min(b.bottom for b in bounds_list) / RES)
    maxx = RES * math.ceil(max(b.right for b in bounds_list) / RES)
    maxy = RES * math.ceil(max(b.top for b in bounds_list) / RES)
    width, height = int(round((maxx-minx)/RES)), int(round((maxy-miny)/RES))
    transform = rasterio.transform.from_origin(minx, maxy, RES, RES)
    print(f'Grid mosaik: {width} x {height} piksel (20 m)', flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_name(OUT.stem + '_sementara.tif')
    if tmp.exists():
        tmp.unlink()

    profile = dict(driver='GTiff', width=width, height=height, count=1, dtype='uint8',
                   crs=CRS_TARGET, transform=transform, nodata=0,
                   tiled=True, blockxsize=512, blockysize=512,
                   compress='deflate', predictor=1, BIGTIFF='IF_SAFER')
    total_written = 0
    try:
        with rasterio.open(tmp, 'w', **profile) as dst:
            dst.write_colormap(1, COLORMAP)
        with rasterio.open(tmp, 'r+') as dst:
            # Setiap blok ditaruh ke grid mosaik yang sama (nearest-neighbor untuk label kelas).
            for i, path in enumerate(rasters, 1):
                with rasterio.open(path) as src:
                    bb = src.bounds
                    bounds_window = from_bounds(bb.left, bb.bottom, bb.right, bb.top, transform=transform)
                    col0 = max(0, math.floor(bounds_window.col_off))
                    row0 = max(0, math.floor(bounds_window.row_off))
                    col1 = min(width, math.ceil(bounds_window.col_off + bounds_window.width))
                    row1 = min(height, math.ceil(bounds_window.row_off + bounds_window.height))
                    if col1 <= col0 or row1 <= row0:
                        continue
                    with WarpedVRT(src, crs=CRS_TARGET, transform=transform,
                                   width=width, height=height,
                                   src_nodata=0, nodata=0,
                                   resampling=Resampling.nearest) as vrt:
                        for row in range(row0, row1, 512):
                            h = min(512, row1-row)
                            for col in range(col0, col1, 512):
                                w = min(512, col1-col)
                                win = Window(col, row, w, h)
                                vals = vrt.read(1, window=win)
                                if not np.any(vals):
                                    continue
                                # Potong ke poligon batas Jawa Timur; 0 = di luar wilayah.
                                mask = geometry_mask(boundary, out_shape=(h,w),
                                                     transform=window_transform(win, transform),
                                                     invert=True, all_touched=False)
                                if not mask.any():
                                    continue
                                vals[~mask] = 0
                                if not vals.any():
                                    continue
                                current = dst.read(1, window=win)
                                fill = (current == 0) & (vals != 0)
                                if fill.any():
                                    current[fill] = vals[fill]
                                    dst.write(current, 1, window=win)
                                    total_written += int(fill.sum())
                if i % 10 == 0 or i == len(rasters):
                    print(f'[MOSAIK] {i}/{len(rasters)} raster | {total_written:,} piksel ditulis', flush=True)

        tmp.replace(OUT)
    except Exception:
        print(f'Proses gagal. File sementara (jika ada): {tmp}', flush=True)
        raise

    counts = Counter()
    with rasterio.open(OUT) as src:
        for _, win in src.block_windows(1):
            vals = src.read(1, window=win)
            bc = np.bincount(vals.ravel(), minlength=6)
            for k in range(6):
                counts[str(k)] += int(bc[k])
    classes = {}
    for key,name in CLASS_NAMES.items():
        pixels = counts[key]
        classes[name] = {'kode': int(key), 'piksel': pixels, 'luas_km2': round(pixels * RES**2 / 1e6, 3)}
    report = {'raster_masuk':len(rasters), 'hasil':str(OUT), 'crs':str(CRS_TARGET),
              'resolusi_meter':RES, 'ukuran_piksel':[width,height],
              'luas_terklasifikasi_km2':round(sum(counts[str(k)] for k in range(1,6))*RES**2/1e6,3),
              'kelas':classes, 'nodata_pixels':counts['0'],
              'catatan':'Kelas 0 mencakup luar provinsi dan piksel tanpa data. Luas adalah luas planar EPSG:32749.'}
    REPORT.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print('\nSELESAI — mosaik seluruh Jawa Timur!',flush=True)
    print('Raster:', OUT, flush=True)
    print('Laporan:',REPORT,flush=True)
    for name, info in classes.items():
        print(f'  {name:<12}: {info["luas_km2"]:,.2f} km2',flush=True)

if __name__ == '__main__':
    main()
