"""Prediksi Random Forest 19 fitur dari raster Sentinel-2 14 band (grid 100 m).
Jalankan dari root: py -3.9 scripts/14_prediksi_sampling_poligon_19fitur.py
File model tetap dibaca dari nama lama (17fitur), karena isinya sudah 19 fitur.
Output tidak menimpa raster klasifikasi lama.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from PIL import Image

BASE = Path(__file__).resolve().parent.parent
RASTER_PATH = BASE / 'data/prediksi/sentinel2_jatim_14fitur_100m.tif'
MODEL_PATH = BASE / 'outputs/models/random_forest_model_mangrove_17fitur.joblib'
OUTPUT_DIR = BASE / 'outputs/maps'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_TIF = OUTPUT_DIR / 'klasifikasi_rf_jatim_mangrove_19fitur.tif'
OUTPUT_PNG = OUTPUT_DIR / 'klasifikasi_rf_jatim_mangrove_19fitur.png'
OUTPUT_JSON = OUTPUT_DIR / 'statistik_klasifikasi_mangrove_19fitur.json'

BANDS = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']
FEATURES_14 = BANDS + ['NDVI', 'NDMI', 'MNDWI', 'NDBI']
FEATURES_19 = FEATURES_14 + ['NDWI', 'MVI', 'REMI', 'MANGROVE_8A_11', 'VEG_8A_4']
CLASSES = ['build up', 'farm land', 'water body', 'forest', 'mangrove']
KODE_KELAS = {'build up': 1, 'farm land': 2, 'water body': 3, 'forest': 4, 'mangrove': 5}
WARNA = {
    0: (0, 0, 0, 0), 1: (128, 128, 128, 255),
    2: (255, 215, 0, 255), 3: (33, 150, 243, 255),
    4: (34, 139, 34, 255), 5: (0, 206, 209, 255),
}
EPS = 1e-6


def safe_ratio(numerator, denominator, absolute=False):
    """Identik dengan safe_ratio pada scripts/02_preprocessing.py."""
    n = np.asarray(numerator, dtype=np.float64)
    d = np.asarray(denominator, dtype=np.float64)
    if absolute:
        n, d = np.abs(n), np.abs(d)
    valid = np.isfinite(n) & np.isfinite(d) & (np.abs(d) > EPS)
    result = np.full(n.shape, np.nan, dtype=np.float64)
    np.divide(n, d, out=result, where=valid)
    return result.astype(np.float32)


def tambah_lima_indeks(x14):
    """Membangun 19 fitur dengan urutan yang sama seperti preprocessing."""
    x = np.asarray(x14, dtype=np.float32)
    b03 = x[:, 1].astype(np.float64)
    b04 = x[:, 2].astype(np.float64)
    b05 = x[:, 3].astype(np.float64)
    b08 = x[:, 6].astype(np.float64)
    b8a = x[:, 7].astype(np.float64)
    b11 = x[:, 8].astype(np.float64)
    ndwi = safe_ratio(b03 - b08, b03 + b08)
    mvi = safe_ratio(b08 - b03, b11 - b03, absolute=True)
    remi = safe_ratio(b05 - b04, b11 - b03)
    mangrove_8a_11 = safe_ratio(b8a - b11, b8a + b11)
    veg_8a_4 = safe_ratio(b8a - b04, b8a + b04)
    return np.column_stack((x, ndwi, mvi, remi, mangrove_8a_11, veg_8a_4)).astype(np.float32)


def main():
    print('=' * 65)
    print('PREDIKSI JAWA TIMUR - RANDOM FOREST 19 FITUR')
    print('=' * 65)
    for path in (RASTER_PATH, MODEL_PATH):
        if not path.is_file():
            raise FileNotFoundError(f'File tidak ditemukan: {path}')

    paket = joblib.load(MODEL_PATH)
    model = paket['model']
    if list(paket['features']) != FEATURES_19:
        raise ValueError(f'Urutan fitur model tidak sesuai.\nModel: {paket["features"]}\nScript: {FEATURES_19}')
    if model.n_features_in_ != len(FEATURES_19):
        raise ValueError('Jumlah fitur model tidak sama dengan 19.')
    if set(model.classes_) != set(CLASSES):
        raise ValueError(f'Kelas model tidak sesuai: {list(model.classes_)}')
    print('Model:', MODEL_PATH.name)
    print('Jumlah fitur:', len(FEATURES_19))
    print('Urutan fitur:', ', '.join(FEATURES_19))

    jumlah_kelas = np.zeros(6, dtype=np.int64)
    with rasterio.open(RASTER_PATH) as src:
        if src.count != 14 or list(src.descriptions) != FEATURES_14:
            raise ValueError(f'Raster harus mempunyai 14 band dengan urutan {FEATURES_14}. Ditemukan: {list(src.descriptions)}')
        print('Ukuran:', src.width, 'x', src.height)
        print('CRS:', src.crs, 'Resolusi:', src.res)
        profile = src.profile.copy()
        profile.update(driver='GTiff', count=1, dtype='uint8', nodata=0,
                       compress='deflate', tiled=True, blockxsize=256, blockysize=256)
        block_size = 512
        total_blok = (((src.height + block_size - 1) // block_size)
                      * ((src.width + block_size - 1) // block_size))
        proses = 0

        with rasterio.open(OUTPUT_TIF, 'w', **profile) as dst:
            dst.write_colormap(1, WARNA)
            dst.set_band_description(1, 'RF 19 fitur: 0=NoData, 1=Built-up, 2=Farm land, 3=Water body, 4=Forest, 5=Mangrove')
            for row in range(0, src.height, block_size):
                for col in range(0, src.width, block_size):
                    h = min(block_size, src.height - row)
                    w = min(block_size, src.width - col)
                    window = Window(col, row, w, h)
                    data = src.read(window=window, out_dtype='float32')
                    x14 = np.moveaxis(data, 0, -1).reshape(-1, 14)
                    valid = np.isfinite(x14).all(axis=1)
                    valid &= (x14[:, :10] > 0).all(axis=1)
                    valid &= src.read_masks(1, window=window).reshape(-1) > 0
                    hasil = np.zeros(x14.shape[0], dtype=np.uint8)
                    if np.any(valid):
                        indices = np.flatnonzero(valid)
                        x19 = tambah_lima_indeks(x14[indices])
                        finite19 = np.isfinite(x19).all(axis=1)
                        if np.any(finite19):
                            prediksi = model.predict(x19[finite19])
                            kode = np.array([KODE_KELAS[str(label)] for label in prediksi], dtype=np.uint8)
                            hasil[indices[finite19]] = kode
                            jumlah_kelas += np.bincount(kode, minlength=6)
                    dst.write(hasil.reshape(h, w), 1, window=window)
                    proses += 1
                    if proses % 10 == 0 or proses == total_blok:
                        print(f'Blok {proses}/{total_blok} selesai', flush=True)

    print('\nMembuat PNG preview...')
    with rasterio.open(OUTPUT_TIF) as src:
        max_size = 2200
        faktor = max(src.width / max_size, src.height / max_size, 1)
        new_width = max(1, int(src.width / faktor))
        new_height = max(1, int(src.height / faktor))
        kelas_png = src.read(1, out_shape=(new_height, new_width), resampling=Resampling.nearest)
    rgba = np.zeros((new_height, new_width, 4), dtype=np.uint8)
    for kode, warna in WARNA.items():
        rgba[kelas_png == kode] = warna
    Image.fromarray(rgba).save(OUTPUT_PNG)

    statistik = {nama: int(jumlah_kelas[KODE_KELAS[nama]]) for nama in CLASSES}
    OUTPUT_JSON.write_text(json.dumps({
        'model': MODEL_PATH.name,
        'jumlah_fitur': 19,
        'fitur': FEATURES_19,
        'jumlah_piksel_valid': int(jumlah_kelas[1:].sum()),
        'jumlah_piksel_per_kelas': statistik,
        'keterangan': 'Belum divalidasi secara spasial; prediksi resolusi asli 100 meter.'
    }, indent=2, ensure_ascii=False), encoding='utf-8')

    print('\n' + '=' * 65)
    print('PREDIKSI 19 FITUR SELESAI')
    for nama, jumlah in statistik.items():
        print(f'{nama}: {jumlah:,} piksel')
    print('GeoTIFF:', OUTPUT_TIF)
    print('Preview:', OUTPUT_PNG)
    print('Statistik:', OUTPUT_JSON)
    print('Model dan raster klasifikasi lama tidak diubah.')


if __name__ == '__main__':
    main()
