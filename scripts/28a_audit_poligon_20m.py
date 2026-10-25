
"""
TAHAP 28A - AUDIT DAN PEMBAGIAN POLIGON SAMPEL
Untuk klasifikasi Sentinel-2 20 meter Jawa Timur.

Input:
data/geojson/GeoJSON_302_BuiltUp_Dilengkapi.geojson

Output:
outputs/reports/tahap28a_sampel.csv
outputs/reports/tahap28a_blok.csv
outputs/reports/tahap28a_ringkasan.json

Jalankan:
py -3.9 scripts/28a_audit_poligon_20m.py
"""

from pathlib import Path
from collections import Counter, defaultdict
import csv
import json
import math

import rasterio
from rasterio.warp import transform_geom
from shapely.geometry import shape, box


BASE = Path(__file__).resolve().parent.parent

GEOJSON = (
    BASE / "data/geojson"
    / "GeoJSON_302_BuiltUp_Dilengkapi.geojson"
)

OUT = BASE / "outputs/reports"
OUT.mkdir(parents=True, exist_ok=True)

CSV_SAMPEL = OUT / "tahap28a_sampel.csv"
CSV_BLOK = OUT / "tahap28a_blok.csv"
JSON_REPORT = OUT / "tahap28a_ringkasan.json"

# Blok pengolahan awal 20 x 20 kilometer.
UKURAN_BLOK = 20000

KELAS = {
    "build up": 1,
    "farm land": 2,
    "water body": 3,
    "forest": 4,
    "mangrove": 5
}

if not GEOJSON.exists():
    raise FileNotFoundError(GEOJSON)

print("=" * 65)
print("TAHAP 28A - AUDIT 335 POLIGON JAWA TIMUR")
print("=" * 65)

with GEOJSON.open("r", encoding="utf-8") as f:
    geojson = json.load(f)

features = geojson.get("features", [])

if not features:
    raise ValueError("GeoJSON tidak memiliki features.")

# GeoJSON standar menggunakan koordinat EPSG:4326.
# Jika file menyatakan CRS berbeda, hentikan agar
# geometri tidak salah diproyeksikan.
crs_info = geojson.get("crs")

if crs_info:
    crs_name = (
        crs_info.get("properties", {})
        .get("name", "")
        .upper()
    )
    if "4326" not in crs_name and "CRS84" not in crs_name:
        raise ValueError(
            "GeoJSON menyatakan CRS selain EPSG:4326. "
            "Periksa CRS sumber sebelum melanjutkan."
        )

rekap = Counter()
rows = []
blok = defaultdict(set)
masalah = []

for nomor, ft in enumerate(features, 1):

    props = ft.get("properties") or {}
    label = str(
        props.get("class", "")
    ).strip().lower()

    if label not in KELAS:
        masalah.append(
            f"Poligon {nomor}: kelas tidak dikenal '{label}'"
        )
        continue

    geometry = ft.get("geometry")

    if not geometry:
        masalah.append(
            f"Poligon {nomor}: geometry kosong"
        )
        continue

    try:
        # Ubah ke proyeksi meter untuk pembagian blok.
        projected = transform_geom(
            "EPSG:4326",
            "EPSG:32749",
            geometry
        )

        geom = shape(projected)

        if geom.is_empty or not geom.is_valid:
            masalah.append(
                f"Poligon {nomor}: geometri kosong/tidak valid"
            )
            continue

        if geom.geom_type not in (
            "Polygon", "MultiPolygon"
        ):
            masalah.append(
                f"Poligon {nomor}: bukan poligon"
            )
            continue

        centroid = geom.centroid
        xmin, ymin, xmax, ymax = geom.bounds

        rekap[label] += 1

        rows.append({
            "id_sampel": nomor,
            "kelas": label,
            "kode": KELAS[label],
            "luas_m2": round(geom.area, 2),
            "centroid_x": round(centroid.x, 2),
            "centroid_y": round(centroid.y, 2)
        })

        # Cari setiap blok yang benar-benar
        # beririsan dengan geometri.
        ix_min = math.floor(xmin / UKURAN_BLOK)
        ix_max = math.floor(xmax / UKURAN_BLOK)
        iy_min = math.floor(ymin / UKURAN_BLOK)
        iy_max = math.floor(ymax / UKURAN_BLOK)

        for ix in range(ix_min, ix_max + 1):
            for iy in range(iy_min, iy_max + 1):

                kiri = ix * UKURAN_BLOK
                bawah = iy * UKURAN_BLOK

                petak = box(
                    kiri,
                    bawah,
                    kiri + UKURAN_BLOK,
                    bawah + UKURAN_BLOK
                )

                if geom.intersects(petak):
                    blok[(ix, iy)].add(nomor)

    except Exception as exc:
        masalah.append(
            f"Poligon {nomor}: {str(exc)[:150]}"
        )

if not rows:
    raise RuntimeError("Tidak ada poligon valid.")

with CSV_SAMPEL.open(
    "w", newline="", encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(rows[0].keys())
    )

    writer.writeheader()
    writer.writerows(rows)

rows_blok = []

for (ix, iy), ids in sorted(blok.items()):

    rows_blok.append({
        "blok_id": f"B_{ix}_{iy}",
        "xmin_utm": ix * UKURAN_BLOK,
        "ymin_utm": iy * UKURAN_BLOK,
        "xmax_utm": (ix + 1) * UKURAN_BLOK,
        "ymax_utm": (iy + 1) * UKURAN_BLOK,
        "jumlah_poligon": len(ids),
        "id_poligon": ",".join(
            str(n) for n in sorted(ids)
        )
    })

with CSV_BLOK.open(
    "w", newline="", encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(rows_blok[0].keys())
    )

    writer.writeheader()
    writer.writerows(rows_blok)

laporan = {
    "jumlah_features_geojson": len(features),
    "jumlah_poligon_valid": len(rows),
    "jumlah_poligon_bermasalah": len(masalah),
    "jumlah_per_kelas": dict(rekap),
    "jumlah_blok_20km": len(rows_blok),
    "ukuran_blok_meter": UKURAN_BLOK,
    "crs_blok": "EPSG:32749",
    "catatan": (
        "Blok hanya untuk merencanakan pengambilan "
        "citra di sekitar sampel. Belum mewakili "
        "seluruh area klasifikasi Jawa Timur."
    ),
    "masalah": masalah
}

with JSON_REPORT.open(
    "w", encoding="utf-8"
) as f:
    json.dump(
        laporan,
        f,
        indent=2,
        ensure_ascii=False
    )

print("\nJumlah GeoJSON:", len(features))
print("Poligon valid:", len(rows))
print("Poligon bermasalah:", len(masalah))

print("\nJUMLAH SAMPEL PER KELAS")

for nama, kode in KELAS.items():
    print(
        f"{nama:12s}: {rekap[nama]}"
    )

print("\nJumlah blok 20 km:", len(rows_blok))

if masalah:
    print("\nPERINGATAN - POLIGON BERMASALAH")
    for m in masalah[:15]:
        print("-", m)

print("\nOUTPUT")
print("Sampel :", CSV_SAMPEL)
print("Blok   :", CSV_BLOK)
print("Laporan:", JSON_REPORT)

print("\nTAHAP 28A SELESAI")
print("Belum mengunduh citra atau melatih model.")
