
"""
TAHAP 28F - TAMBAH DUA BLOK MANGROVE
Tidak mengubah hasil blok B_32_459.

Jalankan:
py -3.9 scripts/28f_tambah_blok_mangrove.py
"""

from pathlib import Path
import csv
import json
import subprocess
import sys
import rasterio
from rasterio.warp import transform_bounds

BASE = Path(__file__).resolve().parent.parent
SCRIPTS = BASE / "scripts"
REPORTS = BASE / "outputs/reports"
GEOJSON_DIR = BASE / "data/geojson"
DATASETS = BASE / "outputs/datasets"

PIPELINE27 = SCRIPTS / "27_komposit_multitanggal_20m.py"
EKSTRAKSI28D = SCRIPTS / "28d_ekstraksi_sampel_20m.py"

BLOK_CSV = REPORTS / "tahap28a_blok.csv"
GEOJSON_ASLI = (
    GEOJSON_DIR / "GeoJSON_302_BuiltUp_Dilengkapi.geojson"
)

TARGET = ["B_28_459", "B_28_456"]

with BLOK_CSV.open("r", encoding="utf-8-sig") as f:
    rows = {r["blok_id"]: r for r in csv.DictReader(f)}

with GEOJSON_ASLI.open("r", encoding="utf-8") as f:
    geojson = json.load(f)

features = geojson["features"]

kode27 = PIPELINE27.read_text(encoding="utf-8")
kode28 = EKSTRAKSI28D.read_text(encoding="utf-8")

old_out = 'OUT = BASE / "data/prediksi/sentinel2_20m_uji"'
old_bbox = 'BBOX = [112.65, -7.30, 112.85, -7.10]'

old_config = (
    'KONFIG = BASE / "outputs/reports/'
    'tahap28b_blok_terpilih.json"'
)

old_geojson = (
    'GEOJSON = BASE / "data/geojson/'
    'tahap28b_sampel_blok.geojson"'
)

for teks, target_code, nama in [
    (old_out, kode27, "OUT Tahap 27"),
    (old_bbox, kode27, "BBOX Tahap 27"),
    (old_config, kode28, "KONFIG Tahap 28D"),
    (old_geojson, kode28, "GEOJSON Tahap 28D"),
]:
    if target_code.count(teks) != 1:
        raise RuntimeError(
            f"Format {nama} berubah. "
            "Periksa script sumber sebelum menjalankan."
        )

for blok_id in TARGET:
    if blok_id not in rows:
        raise ValueError(f"Blok tidak ditemukan: {blok_id}")

    print("\n" + "=" * 65, flush=True)
    print("MEMPROSES BLOK:", blok_id, flush=True)
    print("=" * 65, flush=True)

    row = rows[blok_id]

    bbox_utm = [
        float(row["xmin_utm"]),
        float(row["ymin_utm"]),
        float(row["xmax_utm"]),
        float(row["ymax_utm"]),
    ]

    bbox_ll = list(transform_bounds(
        "EPSG:32749",
        "EPSG:4326",
        *bbox_utm,
        densify_pts=21
    ))

    ids = {
        int(s.strip())
        for s in row["id_poligon"].split(",")
        if s.strip()
    }

    selected = []

    for nomor, ft in enumerate(features, 1):
        if nomor not in ids:
            continue

        selected.append({
            **ft,
            "properties": {
                **(ft.get("properties") or {}),
                "id_sampel_asli": nomor
            }
        })

    if len(selected) != len(ids):
        raise RuntimeError(
            f"Jumlah poligon tidak cocok: {blok_id}"
        )

    file_config = REPORTS / f"tahap28f_{blok_id}.json"
    file_geojson = GEOJSON_DIR / f"tahap28f_{blok_id}.geojson"

    with file_config.open("w", encoding="utf-8") as f:
        json.dump({
            "blok_terpilih": blok_id,
            "bbox_utm": bbox_utm,
            "bbox_lonlat": bbox_ll
        }, f, indent=2)

    with file_geojson.open("w", encoding="utf-8") as f:
        json.dump({
            "type": "FeatureCollection",
            "features": selected
        }, f, ensure_ascii=False)

    folder_output = (
        BASE / "data/prediksi/sentinel2_20m_blok" / blok_id
    )

    raster = (
        folder_output / "komposit_sentinel2_19fitur_20m.tif"
    )

    # 1. Komposit hanya jika belum ada.
    if not raster.exists():

        revisi27 = kode27.replace(
            old_out,
            f'OUT = BASE / "data/prediksi/sentinel2_20m_blok/{blok_id}"',
            1
        ).replace(
            old_bbox,
            f"BBOX = {bbox_ll!r}",
            1
        )

        file27 = SCRIPTS / f"28f_komposit_{blok_id}.py"

        compile(revisi27, str(file27), "exec")
        file27.write_text(revisi27, encoding="utf-8")

        print("Membuat komposit:", blok_id, flush=True)

        subprocess.run(
            [sys.executable, "-u", str(file27)],
            cwd=str(BASE),
            check=True
        )

    else:
        print("Komposit sudah tersedia, dilewati.", flush=True)

    # 2. Ekstraksi fitur.
    revisi28 = kode28.replace(
        old_config,
        f'KONFIG = BASE / "outputs/reports/tahap28f_{blok_id}.json"',
        1
    ).replace(
        old_geojson,
        f'GEOJSON = BASE / "data/geojson/tahap28f_{blok_id}.geojson"',
        1
    )

    file28 = SCRIPTS / f"28f_ekstraksi_{blok_id}.py"

    compile(revisi28, str(file28), "exec")
    file28.write_text(revisi28, encoding="utf-8")

    print("Mengekstraksi sampel:", blok_id, flush=True)

    subprocess.run(
        [sys.executable, "-u", str(file28)],
        cwd=str(BASE),
        check=True
    )

    print("BLOK SELESAI:", blok_id, flush=True)

print("\n" + "=" * 65)
print("DUA BLOK MANGROVE SELESAI")
print("=" * 65)

for blok_id in TARGET:
    path = DATASETS / f"tahap28d_sampel_{blok_id}.csv"
    print("Dataset:", path)

print("\nBelum dilakukan training.")
print("Blok lain masih perlu diproses.")
