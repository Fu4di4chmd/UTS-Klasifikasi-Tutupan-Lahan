
"""
TAHAP 28B - MEMILIH BLOK TRAINING SENTINEL-2 20 M

Input:
- outputs/reports/tahap28a_sampel.csv
- outputs/reports/tahap28a_blok.csv
- data/geojson/GeoJSON_302_BuiltUp_Dilengkapi.geojson

Output:
- outputs/reports/tahap28b_blok_terpilih.json
- data/geojson/tahap28b_sampel_blok.geojson

Jalankan:
py -3.9 scripts/28b_pilih_blok_training.py
"""

from pathlib import Path
from collections import Counter
import csv
import json

from rasterio.warp import transform_bounds


BASE = Path(__file__).resolve().parent.parent
REPORT = BASE / "outputs" / "reports"
GEOJSON_DIR = BASE / "data" / "geojson"

FILE_SAMPEL = REPORT / "tahap28a_sampel.csv"
FILE_BLOK = REPORT / "tahap28a_blok.csv"
FILE_GEOJSON = (
    GEOJSON_DIR / "GeoJSON_302_BuiltUp_Dilengkapi.geojson"
)

OUTPUT_JSON = REPORT / "tahap28b_blok_terpilih.json"
OUTPUT_GEOJSON = GEOJSON_DIR / "tahap28b_sampel_blok.geojson"

KELAS = [
    "build up",
    "farm land",
    "water body",
    "forest",
    "mangrove"
]

print("=" * 65)
print("TAHAP 28B - PEMILIHAN BLOK TRAINING 20 METER")
print("=" * 65)

for file in [FILE_SAMPEL, FILE_BLOK, FILE_GEOJSON]:
    if not file.exists():
        raise FileNotFoundError(file)

with FILE_SAMPEL.open(
    "r", encoding="utf-8-sig", newline=""
) as f:
    sampel = list(csv.DictReader(f))

with FILE_BLOK.open(
    "r", encoding="utf-8-sig", newline=""
) as f:
    blok = list(csv.DictReader(f))

with FILE_GEOJSON.open("r", encoding="utf-8") as f:
    geojson = json.load(f)

features = geojson.get("features", [])

kelas_by_id = {
    int(row["id_sampel"]): row["kelas"]
    for row in sampel
}

kandidat = []

for row in blok:

    ids = [
        int(x.strip())
        for x in row["id_poligon"].split(",")
        if x.strip()
    ]

    hitung = Counter(
        kelas_by_id[i]
        for i in ids
        if i in kelas_by_id
    )

    jumlah_kelas = sum(
        hitung[k] > 0 for k in KELAS
    )

    jumlah_poligon = len(ids)

    # Utamakan keragaman kelas dan jumlah poligon.
    # Mangrove menjadi prioritas tambahan karena
    # kelas tersebut sulit dibedakan dari vegetasi lain.
    skor = (
        jumlah_kelas * 1000
        + min(jumlah_poligon, 500)
        + (100 if hitung["mangrove"] > 0 else 0)
        + (50 if hitung["farm land"] > 0 else 0)
    )

    kandidat.append({
        "blok_id": row["blok_id"],
        "ids": ids,
        "jumlah_kelas": jumlah_kelas,
        "jumlah_poligon": jumlah_poligon,
        "distribusi": dict(hitung),
        "skor": skor,
        "xmin": float(row["xmin_utm"]),
        "ymin": float(row["ymin_utm"]),
        "xmax": float(row["xmax_utm"]),
        "ymax": float(row["ymax_utm"])
    })

if not kandidat:
    raise RuntimeError("Daftar blok kosong.")

kandidat.sort(
    key=lambda x: (
        -x["skor"],
        -x["jumlah_kelas"],
        -x["jumlah_poligon"],
        x["blok_id"]
    )
)

pilih = kandidat[0]

bbox_lonlat = transform_bounds(
    "EPSG:32749",
    "EPSG:4326",
    pilih["xmin"],
    pilih["ymin"],
    pilih["xmax"],
    pilih["ymax"],
    densify_pts=21
)

bbox_lonlat = [
    round(v, 7) for v in bbox_lonlat
]

# ID poligon sama dengan nomor urut fitur
# yang digunakan pada Tahap 28A.
selected_ids = set(pilih["ids"])

selected_features = [
    {
        **ft,
        "properties": {
            **(ft.get("properties") or {}),
            "id_sampel_asli": i
        }
    }
    for i, ft in enumerate(features, start=1)
    if i in selected_ids
]

if len(selected_features) != len(selected_ids):
    raise RuntimeError(
        "ID poligon dan jumlah fitur tidak cocok."
    )

# GeoJSON ini menyimpan geometri sampel asli,
# bukan geometri yang sudah dipotong menurut blok.
subset = {
    "type": "FeatureCollection",
    "features": selected_features
}

with OUTPUT_GEOJSON.open(
    "w", encoding="utf-8"
) as f:
    json.dump(
        subset,
        f,
        ensure_ascii=False,
        indent=2
    )

laporan = {
    "blok_terpilih": pilih["blok_id"],
    "jumlah_poligon": pilih["jumlah_poligon"],
    "jumlah_kelas": pilih["jumlah_kelas"],
    "distribusi_kelas": {
        nama: pilih["distribusi"].get(nama, 0)
        for nama in KELAS
    },
    "bbox_utm": [
        pilih["xmin"],
        pilih["ymin"],
        pilih["xmax"],
        pilih["ymax"]
    ],
    "bbox_lonlat": bbox_lonlat,
    "crs_blok": "EPSG:32749",
    "resolusi_rencana_meter": 20,
    "ids_poligon": sorted(selected_ids),
    "catatan": (
        "Blok ini dipilih berdasarkan persebaran "
        "sampel, bukan berdasarkan akurasi model. "
        "GeoJSON sampel belum dipotong ke batas blok."
    ),
    "lima_kandidat_terbaik": [
        {
            "blok_id": k["blok_id"],
            "jumlah_kelas": k["jumlah_kelas"],
            "jumlah_poligon": k["jumlah_poligon"],
            "distribusi": k["distribusi"]
        }
        for k in kandidat[:5]
    ]
}

with OUTPUT_JSON.open(
    "w", encoding="utf-8"
) as f:
    json.dump(
        laporan,
        f,
        ensure_ascii=False,
        indent=2
    )

print("\nLIMA KANDIDAT TERBAIK")

for k in kandidat[:5]:
    print(
        k["blok_id"],
        "| kelas:", k["jumlah_kelas"],
        "| poligon:", k["jumlah_poligon"],
        "| distribusi:", k["distribusi"]
    )

print("\n" + "=" * 65)
print("BLOK PERCOBAAN TERPILIH")
print("=" * 65)

print("ID blok:", pilih["blok_id"])
print("Jumlah kelas:", pilih["jumlah_kelas"])
print("Jumlah poligon:", pilih["jumlah_poligon"])

print("\nDistribusi sampel:")
for nama in KELAS:
    print(
        f"{nama:12s}:",
        pilih["distribusi"].get(nama, 0)
    )

print("\nBBOX Longitude-Latitude:")
print(bbox_lonlat)

print("\nOutput JSON:", OUTPUT_JSON)
print("Output GeoJSON:", OUTPUT_GEOJSON)

print("\nTAHAP 28B SELESAI")
print("Belum mengunduh citra dan belum training model.")
