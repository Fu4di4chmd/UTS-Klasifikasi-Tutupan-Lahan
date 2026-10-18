# ============================================================
# TAHAP 14 - PREDIKSI RANDOM FOREST EKSPERIMEN
# Sampling Forest merata per poligon - seluruh Jawa Timur
# Output terpisah, tanpa mengubah model / peta utama.
# ============================================================

from pathlib import Path
import joblib
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from PIL import Image

BASE = Path(__file__).resolve().parent.parent
RASTER_PATH = BASE / 'data' / 'prediksi' / 'sentinel2_jatim_14fitur_100m.tif'
MODEL_PATH = BASE / 'outputs' / 'models' / 'random_forest_model_sampling_poligon.joblib'
OUTPUT_DIR = BASE / 'outputs' / 'maps'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_TIF = OUTPUT_DIR / 'klasifikasi_rf_jatim_sampling_poligon.tif'
OUTPUT_PNG = OUTPUT_DIR / 'klasifikasi_rf_jatim_sampling_poligon.png'

FEATURES = [
    'B02', 'B03', 'B04', 'B05', 'B06',
    'B07', 'B08', 'B8A', 'B11', 'B12',
    'NDVI', 'NDMI', 'MNDWI', 'NDBI'
]
CLASSES = ['build up', 'farm land', 'water body', 'forest', 'mangrove']
KODE_KELAS = {'build up': 1, 'farm land': 2, 'water body': 3,
              'forest': 4, 'mangrove': 5}
WARNA = {
    0: (0, 0, 0, 0),
    1: (128, 128, 128, 255),
    2: (255, 215, 0, 255),
    3: (33, 150, 243, 255),
    4: (34, 139, 34, 255),
    5: (0, 206, 209, 255),
}

print('=' * 60)
print('PREDIKSI RF EKSPERIMEN - SAMPLING FOREST PER POLIGON')
print('=' * 60)
for path in [RASTER_PATH, MODEL_PATH]:
    if not path.is_file():
        raise FileNotFoundError(f'File tidak ditemukan: {path}')

paket = joblib.load(MODEL_PATH)
model = paket['model']
model_features = list(paket['features'])
if model_features != FEATURES:
    raise ValueError('Urutan fitur model tidak sesuai dengan raster!')
if set(model.classes_) != set(CLASSES):
    raise ValueError('Daftar kelas model tidak sesuai.')
print('Model eksperimen:', MODEL_PATH.name)
print('Jumlah fitur:', len(FEATURES))
print('Kelas model:', list(model.classes_))

with rasterio.open(RASTER_PATH) as src:
    if src.count != len(FEATURES):
        raise ValueError('Raster harus memiliki 14 band.')
    if list(src.descriptions) != FEATURES:
        raise ValueError('Nama/urutan band GeoTIFF tidak sesuai dengan FEATURES.')
    print('Ukuran raster:', src.width, 'x', src.height)
    print('CRS:', src.crs)
    print('Resolusi:', src.res)

    profile = src.profile.copy()
    profile.update(driver='GTiff', count=1, dtype='uint8', nodata=0,
                   compress='deflate', tiled=True,
                   blockxsize=256, blockysize=256)
    jumlah_kelas = np.zeros(6, dtype=np.int64)
    block_size = 512
    total_blok = (((src.height + block_size - 1) // block_size)
                  * ((src.width + block_size - 1) // block_size))
    proses = 0

    with rasterio.open(OUTPUT_TIF, 'w', **profile) as dst:
        dst.write_colormap(1, WARNA)
        dst.set_band_description(1, 'RF sampling poligon (0=NoData, 1=Built-up, 2=Farmland, 3=Water, 4=Forest, 5=Mangrove)')
        for row in range(0, src.height, block_size):
            for col in range(0, src.width, block_size):
                h = min(block_size, src.height - row)
                w = min(block_size, src.width - col)
                window = Window(col, row, w, h)
                data = src.read(window=window, out_dtype='float32')
                X = np.moveaxis(data, 0, -1).reshape(-1, len(FEATURES))

                # Sama persis dengan filter validitas pada prediksi utama.
                valid = np.isfinite(X).all(axis=1)
                valid &= (X[:, :10] > 0).all(axis=1)
                valid &= src.read_masks(1, window=window).reshape(-1) > 0

                hasil = np.zeros(X.shape[0], dtype=np.uint8)
                if np.any(valid):
                    prediksi = model.predict(X[valid])
                    hasil_valid = np.array(
                        [KODE_KELAS[label] for label in prediksi],
                        dtype=np.uint8,
                    )
                    hasil[valid] = hasil_valid
                    jumlah_kelas += np.bincount(hasil_valid, minlength=6)
                dst.write(hasil.reshape(h, w), 1, window=window)
                proses += 1
                if proses % 10 == 0 or proses == total_blok:
                    print(f'Blok {proses}/{total_blok} selesai')

print('\nMembuat PNG preview...')
with rasterio.open(OUTPUT_TIF) as src:
    max_size = 2200
    faktor = max(src.width / max_size, src.height / max_size, 1)
    new_width = max(1, int(src.width / faktor))
    new_height = max(1, int(src.height / faktor))
    kelas_png = src.read(1, out_shape=(new_height, new_width),
                         resampling=Resampling.nearest)
rgba = np.zeros((new_height, new_width, 4), dtype=np.uint8)
for kode, warna in WARNA.items():
    rgba[kelas_png == kode] = warna
Image.fromarray(rgba).save(OUTPUT_PNG)

print('\n' + '=' * 60)
print('PREDIKSI EKSPERIMEN SELESAI')
print('=' * 60)
for nama in CLASSES:
    kode = KODE_KELAS[nama]
    print(f'{nama}: {jumlah_kelas[kode]:,} piksel')
print('\nGeoTIFF:', OUTPUT_TIF)
print('Preview PNG:', OUTPUT_PNG)
print('Model yang digunakan: Random Forest eksperimen sampling poligon')
print('Training ulang: Tidak')
print('Model dan output utama: Tidak diubah')
