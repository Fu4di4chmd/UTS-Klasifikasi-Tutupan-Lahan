"""Audit independen Water Body RF 19 fitur memakai Dynamic World 2026.
Jalankan dari root: py -3.9 scripts/18_validasi_waterbody_dynamicworld.py
Hasil bersifat diagnostik; TIDAK mengubah klasifikasi RF.
"""
from pathlib import Path
import json
import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling

BASE = Path(__file__).resolve().parent.parent
RF = BASE / 'outputs/maps/klasifikasi_rf_jatim_mangrove_19fitur.tif'
DW = BASE / 'data/prediksi/referensi_water_jatim_100m_2026.tif'
OUT = BASE / 'outputs/maps'
REPORT = BASE / 'outputs/reports'
OUT.mkdir(parents=True, exist_ok=True)
REPORT.mkdir(parents=True, exist_ok=True)
AUDIT = OUT / 'audit_waterbody_rf19_dynamicworld_100m.tif'
JSON = REPORT / 'audit_waterbody_rf19_dynamicworld.json'

# Kategori audit: 0=di luar analisis, 1=RF non-air, 2=RF air & DW >=70,
# 3=RF air & DW 20..69, 4=RF air & DW <20, 5=RF air & DW NoData
CODES = {'non_water': 1, 'water_strong': 2, 'water_uncertain': 3,
         'water_low': 4, 'water_no_reference': 5}

for p in [RF, DW]:
    if not p.is_file():
        raise FileNotFoundError(f'Tidak ditemukan: {p}')

counts = np.zeros(6, dtype=np.int64)
with rasterio.open(RF) as rf, rasterio.open(DW) as dw:
    if rf.count != 1 or dw.count < 1:
        raise ValueError('Raster RF harus satu band; referensi DW minimal satu band.')
    if rf.crs is None or dw.crs is None:
        raise ValueError('CRS raster tidak lengkap.')
    print('RF:', rf.name, '| resolusi:', rf.res)
    print('DW:', dw.name, '| resolusi:', dw.res)
    # Nearest tidak mengubah probabilitas melalui interpolasi; grid diselaraskan ke RF.
    with WarpedVRT(dw, crs=rf.crs, transform=rf.transform,
                   width=rf.width, height=rf.height,
                   resampling=Resampling.nearest, nodata=255) as aligned:
        profile = rf.profile.copy()
        profile.update(driver='GTiff', count=1, dtype='uint8', nodata=0,
                       compress='deflate', tiled=True, blockxsize=256, blockysize=256)
        with rasterio.open(AUDIT, 'w', **profile) as dst:
            dst.set_band_description(1, 'Audit Water Body RF19 vs Dynamic World (bukan klasifikasi baru)')
            dst.write_colormap(1, {0: (0,0,0,0), 1: (140,140,140,255),
                                   2: (0,125,255,255), 3: (255,200,0,255),
                                   4: (235,60,60,255), 5: (155,60,200,255)})
            for _, window in rf.block_windows(1):
                a = rf.read(1, window=window)
                b = aligned.read(1, window=window)
                valid_ref = (b != 255) & np.isfinite(b) & (b <= 100)
                classified = (a >= 1) & (a <= 5)
                water = (a == 3)
                result = np.zeros(a.shape, dtype=np.uint8)
                result[classified & ~water] = 1
                result[water & valid_ref & (b >= 70)] = 2
                result[water & valid_ref & (b >= 20) & (b < 70)] = 3
                result[water & valid_ref & (b < 20)] = 4
                result[water & ~valid_ref] = 5
                counts += np.bincount(result.ravel(), minlength=6)
                dst.write(result, 1, window=window)

report = {
    'jenis': 'Audit probabilitas air Dynamic World terhadap hasil RF 19 fitur',
    'peringatan': 'Referensi bukan ground truth; tidak dilakukan perubahan klasifikasi.',
    'piksel_rf_water_total': int(counts[2:].sum()),
    'rf_water_dw_ge_70': int(counts[2]),
    'rf_water_dw_20_69': int(counts[3]),
    'rf_water_dw_lt_20': int(counts[4]),
    'rf_water_tanpa_referensi': int(counts[5]),
    'piksel_rf_non_water': int(counts[1]),
    'threshold_probabilitas': {'kuat': '>=70', 'belum_pasti': '20-69', 'rendah': '<20'},
    'catatan': 'Nilai Dynamic World dari komposit median yang disampel ke 100m, bukan rata-rata 10m.'
}
JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
print('\n=== HASIL AUDIT WATER BODY RF 19 FITUR ===')
for k, v in report.items():
    if k.startswith('rf_water') or k == 'piksel_rf_water_total':
        print(k, ':', v)
print('Raster audit:', AUDIT)
print('Laporan:', JSON)
print('Peta RF asli TIDAK diubah.')
