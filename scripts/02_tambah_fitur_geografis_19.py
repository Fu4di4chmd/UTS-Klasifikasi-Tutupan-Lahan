"""Menambahkan Elevasi dan Jarak Pantai ke CSV 17 fitur dari GeoJSON terbaru.
Jalankan dari root proyek: py -3.9 scripts/02_tambah_fitur_geografis_19.py
Tidak mengubah CSV/model/peta 17 fitur.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from scipy.ndimage import distance_transform_edt

BASE = Path(__file__).resolve().parent.parent
RASTER = BASE / 'data/prediksi/sentinel2_jatim_training_buffer3km_14fitur_100m.tif'
PRED_RASTER = BASE / 'data/prediksi/sentinel2_jatim_14fitur_100m.tif'
DEM = BASE / 'data/prediksi/dem_jatim_aligned_100m.tif'
COAST = BASE / 'data/coastline/ne_10m_coastline.shp'
GEOJSON_DIR = BASE / 'data/geojson'
INPUT = BASE / 'outputs/reports/eksperimen_mangrove_17fitur'
OUTPUT = BASE / 'outputs/reports/eksperimen_mangrove_19fitur'
GEO_OUT = BASE / 'data/prediksi'
OUTPUT.mkdir(parents=True, exist_ok=True)
FEATURES_14 = ['B02','B03','B04','B05','B06','B07','B08','B8A','B11','B12','NDVI','NDMI','MNDWI','NDBI']
FEATURES_17 = FEATURES_14 + ['NDWI','MVI','REMI']
FEATURES_19 = FEATURES_17 + ['elevasi_m','jarak_pantai_km']
NODATA = -9999.0

for p in [RASTER, PRED_RASTER, DEM, COAST, INPUT/'training_17fitur.csv', INPUT/'testing_17fitur.csv']:
    if not p.is_file():
        raise FileNotFoundError(f'File tidak ditemukan: {p}')
files = list(GEOJSON_DIR.glob('*.geojson'))
if len(files) != 1:
    raise ValueError('Folder data/geojson harus berisi tepat satu file .geojson')
gdf = gpd.read_file(files[0])
if len(gdf) != 305:
    raise ValueError(f'GeoJSON terbaca {len(gdf)} poligon, bukan 305. Periksa file.')
gdf['polygon_id'] = np.arange(1, len(gdf)+1)
train = pd.read_csv(INPUT/'training_17fitur.csv')
test = pd.read_csv(INPUT/'testing_17fitur.csv')
for name, frame in [('training',train),('testing',test)]:
    missing = set(FEATURES_17 + ['polygon_id','label'])-set(frame.columns)
    if missing:
        raise ValueError(f'{name} tidak memiliki kolom: {sorted(missing)}')
if set(train.polygon_id) & set(test.polygon_id):
    raise ValueError('Ada kebocoran polygon_id training/testing')

with rasterio.open(RASTER) as src, rasterio.open(DEM) as dem:
    if src.count != 14 or list(src.descriptions) != FEATURES_14:
        raise ValueError('Urutan fitur pada raster training bukan 14 fitur yang diharapkan')
    if src.crs != dem.crs:
        raise ValueError('CRS DEM berbeda. Script perlu disesuaikan sebelum dijalankan.')
    geo = gdf.to_crs(src.crs)
    ids = rasterize(zip(geo.geometry, geo.polygon_id.astype(int)),
                    out_shape=src.shape, transform=src.transform,
                    fill=0, dtype='int32')
    rows, cols = np.where(ids > 0)
    vals = {}
    valid = np.ones(len(rows), dtype=bool)
    for i, band in enumerate(FEATURES_14, 1):
        a = src.read(i, masked=True)
        v = np.asarray(a[rows,cols].filled(np.nan), dtype=np.float32)
        vals[band] = v
        valid &= np.isfinite(v)
        if i <= 10:
            valid &= v > 0
        print('Ekstraksi',i,'/14:',band)
    # Filter 3 indeks tambahan dengan definisi sama dengan preprocessing 17 fitur.
    def ratio(a,b,absolute=False):
        if absolute:
            a,b = np.abs(a),np.abs(b)
        out = np.full(a.shape, np.nan, dtype=np.float32)
        np.divide(a,b,out=out,where=np.isfinite(a)&np.isfinite(b)&(np.abs(b)>1e-6))
        return out
    ndwi = ratio(vals['B03']-vals['B08'], vals['B03']+vals['B08'])
    mvi = ratio(vals['B08']-vals['B03'], vals['B11']-vals['B03'], True)
    remi = ratio(vals['B05']-vals['B04'], vals['B11']-vals['B03'])
    valid &= np.isfinite(ndwi)&np.isfinite(mvi)&np.isfinite(remi)
    rows,cols = rows[valid],cols[valid]
    pids = ids[rows,cols]
    # Validasi keras: jumlah dan urutan piksel wajib cocok dengan CSV 17 fitur.
    train_mask = np.isin(pids, train.polygon_id.unique())
    test_mask = np.isin(pids, test.polygon_id.unique())
    if np.any(~(train_mask | test_mask)) or np.any(train_mask & test_mask):
        raise ValueError('Ada poligon aktif di raster yang tidak cocok dengan CSV 17 fitur')
    for name, frame, sel in [('TRAIN',train,train_mask),('TEST',test,test_mask)]:
        if sel.sum() != len(frame):
            raise ValueError(f'{name}: jumlah piksel tidak cocok ({sel.sum()} vs {len(frame)}).')
        if not np.array_equal(pids[sel], frame.polygon_id.to_numpy()):
            raise ValueError(f'{name}: urutan polygon_id tidak cocok dengan CSV. Proses dihentikan.')
        for band in ['B03','B04','B08','B11']:
            if not np.allclose(vals[band][valid][sel], frame[band].to_numpy(), atol=1e-4, rtol=1e-5):
                raise ValueError(f'{name}: nilai {band} tidak cocok dengan CSV. Proses dihentikan.')
    print('CSV 17 fitur cocok dengan raster training dan GeoJSON 305 poligon.')

    # Elevasi: DEM dicocokkan berdasarkan georeferensi, BUKAN indeks piksel.
    with WarpedVRT(dem, crs=src.crs, transform=src.transform,
                   width=src.width, height=src.height,
                   resampling=Resampling.nearest, nodata=NODATA) as vrt:
        elev = np.asarray(vrt.read(1, masked=True).filled(NODATA), dtype=np.float32)
    elev[~np.isfinite(elev)] = NODATA

    # Garis pantai Natural Earth diubah ke grid training 100 m.
    coast = gpd.read_file(COAST).to_crs(src.crs)
    bbox = geo.total_bounds
    margin = 50000  # hanya percepat seleksi geometri; bukan batas habitat
    bounds = (src.bounds.left-margin,src.bounds.bottom-margin,
              src.bounds.right+margin,src.bounds.top+margin)
    coast = coast.cx[bounds[0]:bounds[2],bounds[1]:bounds[3]]
    if coast.empty:
        raise ValueError('Tidak ada garis pantai dalam wilayah raster.')
    coastal_pixels = rasterize(((g,1) for g in coast.geometry if not g.is_empty),
                               out_shape=src.shape,transform=src.transform,
                               fill=0,dtype='uint8',all_touched=True)
    if not np.any(coastal_pixels):
        raise ValueError('Garis pantai tidak berpotongan dengan grid raster.')
    # Jarak ke garis pantai, satuan kilometer; resolusi grid 100 m.
    dist = distance_transform_edt(coastal_pixels == 0,
                                  sampling=(abs(src.res[1]),abs(src.res[0]))).astype(np.float32)/1000
    del coastal_pixels

    def save_split(frame, selector, name):
        out = frame.copy()
        out['elevasi_m'] = elev[rows[selector],cols[selector]]
        out['jarak_pantai_km'] = dist[rows[selector],cols[selector]]
        good = (np.isfinite(out[['elevasi_m','jarak_pantai_km']]).all(axis=1)
                & (out.elevasi_m != NODATA))
        n_drop = int((~good).sum())
        out = out.loc[good].copy()
        out.to_csv(OUTPUT/f'{name}_19fitur.csv',index=False)
        print(f'{name}: {len(frame)} -> {len(out)} piksel, DEM tidak tersedia: {n_drop}')
        print(out.label.value_counts().to_string())
        if out.empty:
            raise ValueError(f'{name}: semua piksel kehilangan DEM.')
        return out
    tr = save_split(train,train_mask,'training')
    te = save_split(test,test_mask,'testing')
    (OUTPUT/'fitur_model.txt').write_text('\n'.join(FEATURES_19)+'\n',encoding='utf-8')

    # Simpan fitur spasial untuk prediksi model 19 fitur di grid asli.
    with rasterio.open(PRED_RASTER) as pred:
        if pred.crs != src.crs or pred.res != src.res:
            raise ValueError('CRS atau resolusi raster prediksi berbeda')
        profile = pred.profile.copy()
        profile.update(driver='GTiff',count=2,dtype='float32',nodata=NODATA,
                       compress='deflate',tiled=True,blockxsize=256,blockysize=256)
        # Grid prediksi dipotong secara georeferensi dari grid buffer training.
        row0,col0 = src.index(pred.bounds.left+50,pred.bounds.top-50)
        if not np.isclose(src.transform.c + col0*src.res[0], pred.transform.c) or not np.isclose(src.transform.f + row0*src.transform.e,pred.transform.f):
            raise ValueError('Grid prediksi tidak sejajar dengan grid training.')
        r1,c1 = row0+pred.height,col0+pred.width
        if row0 < 0 or col0 < 0 or r1 > src.height or c1 > src.width:
            raise ValueError('Grid prediksi melampaui raster training.')
        result = GEO_OUT/'fitur_geografis_mangrove_19fitur_prediksi.tif'
        with rasterio.open(result,'w',**profile) as dst:
            dst.write(elev[row0:r1,col0:c1],1)
            dst.write(dist[row0:r1,col0:c1],2)
            dst.set_band_description(1,'elevasi_m')
            dst.set_band_description(2,'jarak_pantai_km')
        print('Raster 2 fitur geografis untuk prediksi:',result)
print('SELESAI. Fitur total:',len(FEATURES_19))
print('CSV 19 fitur:',OUTPUT)
print('JANGAN jalankan training 17 fitur lama tanpa penyesuaian ke 19 fitur.')
