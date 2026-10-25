
from pathlib import Path
import requests
import json

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "data" / "geojson" / "batas_jawa_timur.geojson"
OUT.parent.mkdir(parents=True, exist_ok=True)

API = "https://www.geoboundaries.org/api/current/gbOpen/IDN/ADM1/"

print("Mengambil metadata batas provinsi Indonesia...")

response = requests.get(API, timeout=60)
response.raise_for_status()
metadata = response.json()

url = metadata["gjDownloadURL"]

print("Mengunduh batas provinsi...")
response = requests.get(url, timeout=180)
response.raise_for_status()
geojson = response.json()

if geojson.get("type") != "FeatureCollection":
    raise ValueError("Format data bukan GeoJSON FeatureCollection")

print("Jumlah provinsi dalam sumber:", len(geojson["features"]))

hasil = []

for feature in geojson["features"]:
    props = feature.get("properties", {})

    teks = " ".join(str(v) for v in props.values()).lower()

    if "jawa timur" in teks or "east java" in teks:
        hasil.append(feature)
        print("Provinsi ditemukan:", props)

if len(hasil) != 1:
    raise ValueError(
        f"Harus ditemukan tepat satu wilayah Jawa Timur, "
        f"tetapi ditemukan {len(hasil)}. Periksa nama provinsi."
    )

if hasil[0].get("geometry") is None:
    raise ValueError("Geometri Jawa Timur kosong.")

output = {
    "type": "FeatureCollection",
    "features": hasil
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False)

print("\nBERHASIL")
print("Batas Jawa Timur tersimpan:", OUT)
print("Jenis geometri:", hasil[0]["geometry"]["type"])
print("Sumber: geoBoundaries gbOpen Indonesia ADM1")
