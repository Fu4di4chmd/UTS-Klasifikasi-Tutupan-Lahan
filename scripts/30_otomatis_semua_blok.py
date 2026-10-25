from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from pyproj import Transformer
import argparse
import csv
import re
import shutil
import subprocess
import sys
import time
import json
import rasterio
import numpy as np

BASE = Path(__file__).resolve().parents[1]
SCRIPTS = BASE / 'scripts'
DAFTAR = BASE / 'outputs/reports/30_daftar_blok_seluruh_jatim.csv'
SUMBER_KOMPOSIT = SCRIPTS / '27_komposit_multitanggal_20m.py'
SUMBER_PREDIKSI = SCRIPTS / '30_klasifikasi_jawa_timur.py'
MODEL = BASE / 'outputs/models/random_forest_20m_19fitur.joblib'
KOMPOSIT_DIR = BASE / 'data/prediksi/sentinel2_20m_blok'
PETA_DIR = BASE / 'outputs/maps/klasifikasi_20m'
TEMP_DIR = SCRIPTS / '_otomatis30'
LOG_DIR = BASE / 'outputs/logs/tahap30'
SALAH_DIR = SCRIPTS / 'data/prediksi/sentinel2_20m_blok'


def ganti_satu(teks, pola, pengganti, nama):
    hasil, n = re.subn(pola, lambda m: pengganti, teks, count=1, flags=re.MULTILINE)
    if n != 1:
        raise ValueError(f'Konfigurasi {nama} tidak ditemukan pada script sumber.')
    return hasil


def betulkan_base(teks):
    # Menangani BASE = Path(__file__).resolve().parent.parent
    # serta BASE = Path(__file__).resolve().parents[1].
    pola = r'^BASE\s*=\s*Path\(__file__\)\.resolve\(\)\.(?:parent\.parent|parents\[1\])\s*$'
    return ganti_satu(teks, pola, f'BASE = Path({str(BASE)!r})', 'BASE')


def bbox_lonlat(row):
    xmin = float(row['xmin_utm'])
    ymin = float(row['ymin_utm'])
    xmax = float(row['xmax_utm'])
    ymax = float(row['ymax_utm'])
    tf = Transformer.from_crs('EPSG:32749', 'EPSG:4326', always_xy=True)
    pts = [tf.transform(x, y) for x in (xmin, xmax) for y in (ymin, ymax)]
    return [min(p[0] for p in pts), min(p[1] for p in pts),
            max(p[0] for p in pts), max(p[1] for p in pts)]


def lokasi(blok):
    return (KOMPOSIT_DIR / blok / 'komposit_sentinel2_19fitur_20m.tif',
            PETA_DIR / f'klasifikasi_rf_{blok}_20m.tif')


def cek_raster(path, jumlah_band):
    if not path.is_file():
        return False
    try:
        with rasterio.open(path) as src:
            if (src.count != jumlah_band or src.width < 1 or src.height < 1
                    or src.crs is None or
                    abs(src.res[0] - 20) > .01 or abs(src.res[1] - 20) > .01):
                return False
            # Cek mask / isi representatif, tanpa membaca seluruh raster.
            sample = src.read(1, window=((0, min(32, src.height)),
                                         (0, min(32, src.width))), masked=True)
            if sample.size == 0:
                return False
        return True
    except Exception:
        return False


def selamatkan_file():
    if not SALAH_DIR.is_dir():
        return
    print('Memeriksa hasil komposit yang tersimpan di folder scripts/data...', flush=True)
    for folder in SALAH_DIR.iterdir():
        if not folder.is_dir():
            continue
        nama = folder.name
        # Salin hanya raster komposit yang sudah lengkap/terbaca.
        src19 = folder / 'komposit_sentinel2_19fitur_20m.tif'
        src10 = folder / 'komposit_sentinel2_10band_20m.tif'
        if not cek_raster(src19, 19) or not cek_raster(src10, 10):
            print(f'[LEWATI CACHE] {nama}: komposit tidak lengkap', flush=True)
            continue
        tujuan = KOMPOSIT_DIR / nama
        tujuan.mkdir(parents=True, exist_ok=True)
        for sumber in (src10, src19, folder / 'laporan_komposit_tahap27.json'):
            if sumber.is_file():
                dst = tujuan / sumber.name
                if not dst.exists():
                    shutil.copy2(sumber, dst)
                    print(f'[SALIN CACHE] {nama}: {sumber.name}', flush=True)


def jalankan(script, log, mode='w'):
    with log.open(mode, encoding='utf-8') as fp:
        p = subprocess.run([sys.executable, '-u', str(script)],
                           cwd=str(BASE), stdout=fp, stderr=subprocess.STDOUT)
    if p.returncode:
        raise RuntimeError(f'Exit code {p.returncode}; lihat log {log}')


def proses_blok(row):
    blok = row['blok_id']
    fitur, peta = lokasi(blok)
    log = LOG_DIR / f'{blok}.log'
    skrip_komposit = TEMP_DIR / f'komposit_{blok}.py'
    skrip_prediksi = TEMP_DIR / f'prediksi_{blok}.py'
    awal = time.monotonic()
    try:
        if cek_raster(peta, 1):
            return blok, 'lewati', 0

        if not cek_raster(fitur, 19):
            print(f'[KOMPOSIT] {blok}', flush=True)
            teks = betulkan_base(SUMBER_KOMPOSIT.read_text(encoding='utf-8-sig'))
            teks = ganti_satu(
                teks,
                r'^OUT\s*=\s*BASE\s*/\s*["\']data/prediksi/sentinel2_20m_uji["\']',
                f'OUT = BASE / "data/prediksi/sentinel2_20m_blok/{blok}"',
                'OUT'
            )
            teks = ganti_satu(teks, r'^BBOX\s*=\s*\[[^\]\n]+\]',
                               f'BBOX = {repr(bbox_lonlat(row))}', 'BBOX')
            compile(teks, str(skrip_komposit), 'exec')
            skrip_komposit.write_text(teks, encoding='utf-8')
            jalankan(skrip_komposit, log, 'w')
            if not cek_raster(fitur, 19):
                raise RuntimeError('Raster komposit 19 fitur belum valid')
        else:
            print(f'[CACHE] {blok}: komposit tersedia', flush=True)

        print(f'[PREDIKSI] {blok}', flush=True)
        teks = betulkan_base(SUMBER_PREDIKSI.read_text(encoding='utf-8-sig'))
        teks = ganti_satu(teks, r'^BLOK\s*=\s*\[[\s\S]*?\]',
                           f'BLOK = ["{blok}"]', 'BLOK')
        teks = teks.replace('laporan_klasifikasi_5blok.json',
                            f'laporan_klasifikasi_{blok}.json')
        compile(teks, str(skrip_prediksi), 'exec')
        skrip_prediksi.write_text(teks, encoding='utf-8')
        jalankan(skrip_prediksi, log, 'a')
        if not cek_raster(peta, 1):
            raise RuntimeError('Raster prediksi belum valid')
        menit = (time.monotonic() - awal) / 60
        print(f'[SELESAI] {blok} ({menit:.1f} menit)', flush=True)
        return blok, 'berhasil', menit
    except Exception as exc:
        menit = (time.monotonic() - awal) / 60
        print(f'[GAGAL] {blok}: {exc}', flush=True)
        return blok, 'gagal', menit
    finally:
        for fp in (skrip_komposit, skrip_prediksi):
            try:
                fp.unlink(missing_ok=True)
            except OSError:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=195)
    ap.add_argument('--workers', type=int, default=2)
    args = ap.parse_args()
    if args.limit < 1 or args.workers not in (1, 2, 3):
        ap.error('--limit >=1 dan --workers 1, 2, atau 3')

    for p in (DAFTAR, SUMBER_KOMPOSIT, SUMBER_PREDIKSI, MODEL):
        if not p.is_file():
            raise FileNotFoundError(p)
    for directory in (KOMPOSIT_DIR, PETA_DIR, TEMP_DIR, LOG_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    selamatkan_file()

    with DAFTAR.open(encoding='utf-8-sig', newline='') as fp:
        daftar = list(csv.DictReader(fp))
    if not daftar:
        raise ValueError('CSV daftar blok kosong')
    daftar.sort(key=lambda r: float(r['luas_irisan_km2']), reverse=True)
    tersedia = sum(cek_raster(lokasi(r['blok_id'])[1], 1) for r in daftar)
    belum = [r for r in daftar if not cek_raster(lokasi(r['blok_id'])[1], 1)]
    target = belum[:args.limit]

    print('\n' + '=' * 65)
    print('TAHAP 30 - OTOMATISASI JAWA TIMUR (BASE DIPERBAIKI)')
    print('=' * 65)
    print(f'Total blok: {len(daftar)} | Selesai: {tersedia} | Sisa: {len(belum)}')
    print(f'Diproses: {len(target)} | Workers: {args.workers}')
    print(f'Output: {PETA_DIR}')
    print('=' * 65, flush=True)

    hasil = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(proses_blok, r) for r in target]
        for task in as_completed(futures):
            hasil.append(task.result())
            print(f'[PROGRES] {len(hasil)}/{len(target)}', flush=True)

    rekap = LOG_DIR / 'ringkasan_proses.csv'
    with rekap.open('w', encoding='utf-8', newline='') as fp:
        w = csv.writer(fp)
        w.writerow(['blok', 'status', 'menit'])
        w.writerows(hasil)
    total_akhir = sum(cek_raster(lokasi(r['blok_id'])[1], 1) for r in daftar)
    gagal = [x[0] for x in hasil if x[1] == 'gagal']
    print('\nRINGKASAN TAHAP 30')
    print(f'Berhasil: {sum(x[1] == "berhasil" for x in hasil)}')
    print(f'Gagal: {len(gagal)}')
    print(f'Total raster: {total_akhir}/{len(daftar)}')
    print(f'Log: {LOG_DIR}')
    if gagal:
        print('Blok gagal:', ', '.join(gagal))

if __name__ == '__main__':
    main()
