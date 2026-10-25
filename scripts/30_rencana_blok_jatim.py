
from pathlib import Path
from math import floor, ceil
from collections import Counter
import json
import csv

from shapely.geometry import shape, box
from shapely.ops import transform
from pyproj import Transformer

BASE = Path(__file__).resolve().parents[1]

BATAS = BASE / "data/geojson/batas_jawa_timur.geojson"
OUTPUT = BASE / "outputs/reports"
RASTER_DIR = BASE / "data/prediksi/sentinel2_20m_blok"

OUTPUT.mkdir(parents=True, exist_ok=True)

UKURAN_BLOK = 20000  # 20 km dalam meter

with open(BATAS, encoding="utf-8") as f:
    geojson = json.load(f)

geometri = [
    shape(feature["geometry"])
    for feature in geojson["features"]
    if feature.get("geometry")
]

if not geometri:
    raise ValueError("Geometri batas Jawa Timur tidak ditemukan.")

from shapely.ops import unary_union

wilayah_wgs84 = unary_union(geometri)

transformer = Transformer.from_crs(
    "EPSG:4326",
    "EPSG:32749",
    always_xy=True
)

wilayah = transform(transformer.transform, wilayah_wgs84)

xmin, ymin, xmax, ymax = wilayah.bounds

kolom_awal = floor(xmin / UKURAN_BLOK)
kolom_akhir = ceil(xmax / UKURAN_BLOK)
baris_awal = floor(ymin / UKURAN_BLOK)
baris_akhir = ceil(ymax / UKURAN_BLOK)

hasil = []
status = Counter()

for ix in range(kolom_awal, kolom_akhir):
    for iy in range(baris_awal, baris_akhir):

        x0 = ix * UKURAN_BLOK
        y0 = iy * UKURAN_BLOK

        kotak = box(
            x0, y0,
            x0 + UKURAN_BLOK,
            y0 + UKURAN_BLOK
        )

        # Hindari blok yang hanya bersentuhan di garis batas.
        irisan = kotak.intersection(wilayah)

        if irisan.is_empty or irisan.area <= 0:
            continue

        blok_id = f"B_{ix}_{iy}"

        raster = (
            RASTER_DIR
            / blok_id
            / "komposit_sentinel2_19fitur_20m.tif"
        )

        tersedia = raster.is_file()
        kondisi = "tersedia" if tersedia else "belum"

        status[kondisi] += 1

        hasil.append({
            "blok_id": blok_id,
            "xmin_utm": x0,
            "ymin_utm": y0,
            "xmax_utm": x0 + UKURAN_BLOK,
            "ymax_utm": y0 + UKURAN_BLOK,
            "luas_irisan_km2": round(
                irisan.area / 1_000_000, 3
            ),
            "komposit_tersedia": tersedia
        })

hasil.sort(key=lambda x: x["blok_id"])

csv_path = OUTPUT / "30_daftar_blok_seluruh_jatim.csv"
json_path = OUTPUT / "30_rencana_blok_seluruh_jatim.json"

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(hasil[0]))
    writer.writeheader()
    writer.writerows(hasil)

laporan = {
    "crs": "EPSG:32749",
    "ukuran_blok_meter": UKURAN_BLOK,
    "jumlah_blok": len(hasil),
    "komposit_tersedia": status["tersedia"],
    "komposit_belum": status["belum"],
    "luas_geometri_km2": round(
        wilayah.area / 1_000_000, 2
    ),
    "blok": hasil
}

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(laporan, f, indent=2)

print("=" * 60)
print("TAHAP 30 - RENCANA BLOK SELURUH JAWA TIMUR")
print("=" * 60)
print("CRS                  : EPSG:32749")
print("Ukuran blok          : 20 x 20 km")
print("Jumlah seluruh blok  :", len(hasil))
print("Komposit tersedia    :", status["tersedia"])
print("Komposit belum ada   :", status["belum"])
print("Luas batas (km2)     :", laporan["luas_geometri_km2"])
print("CSV                  :", csv_path)
print("JSON                 :", json_path)

print("\nBLOK YANG SUDAH TERSEDIA:")
for b in hasil:
    if b["komposit_tersedia"]:
        print(" -", b["blok_id"])

print("\nPerencanaan selesai. Belum ada citra yang diunduh.")
