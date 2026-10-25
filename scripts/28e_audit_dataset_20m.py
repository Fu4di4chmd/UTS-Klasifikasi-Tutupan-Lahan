
from pathlib import Path
import json
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "outputs/datasets"

INPUT = DATA / "tahap28d_sampel_B_32_459.csv"
STAT = DATA / "tahap28d_statistik_B_32_459.csv"
OUTPUT = DATA / "tahap28e_audit_B_32_459.json"

FEATURES = [
    "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B11", "B12",
    "NDVI", "NDMI", "MNDWI", "NDBI",
    "NDWI", "MVI", "REMI",
    "MANGROVE_8A_11", "VEG_8A_4"
]

KELAS = [
    "build up", "farm land", "water body",
    "forest", "mangrove"
]

print("=" * 65)
print("TAHAP 28E - AUDIT DATASET TRAINING 20 METER")
print("=" * 65)

df = pd.read_csv(INPUT)
stat = pd.read_csv(STAT)

missing = [x for x in FEATURES if x not in df.columns]
if missing:
    raise ValueError(f"Fitur hilang: {missing}")

jumlah_per_kelas = df["kelas"].value_counts()
poligon_per_kelas = df.groupby("kelas")["id_poligon"].nunique()

tidak_valid = int(
    (~np.isfinite(df[FEATURES].to_numpy(dtype=float))).any(axis=1).sum()
)

duplikat = int(
    df.duplicated(
        subset=["blok_id", "pixel_row", "pixel_col"]
    ).sum()
)

nol = stat.loc[
    stat["piksel_diambil"] == 0,
    ["id_poligon", "kelas"]
].to_dict("records")

print("\nDISTRIBUSI PER KELAS")
print(f"{'Kelas':15} {'Piksel':>10} {'Poligon':>10}")

distribusi = {}

for kelas in KELAS:
    piksel = int(jumlah_per_kelas.get(kelas, 0))
    poligon = int(poligon_per_kelas.get(kelas, 0))

    distribusi[kelas] = {
        "piksel": piksel,
        "poligon_aktif": poligon
    }

    print(f"{kelas:15} {piksel:10} {poligon:10}")

print("\nPEMERIKSAAN DATA")
print("Jumlah baris:", len(df))
print("Jumlah fitur:", len(FEATURES))
print("Baris dengan fitur tidak valid:", tidak_valid)
print("Duplikasi posisi piksel:", duplikat)
print("Poligon tanpa sampel:", nol)

laporan = {
    "blok": "B_32_459",
    "jumlah_baris": int(len(df)),
    "jumlah_fitur": len(FEATURES),
    "distribusi_kelas": distribusi,
    "baris_tidak_valid": tidak_valid,
    "duplikasi_posisi_piksel": duplikat,
    "poligon_tanpa_sampel": nol,
    "siap_training_lima_kelas": False,
    "alasan": (
        "Belum memiliki sampel Mangrove dan "
        "jumlah poligon beberapa kelas sangat sedikit."
    )
}

with OUTPUT.open("w", encoding="utf-8") as f:
    json.dump(laporan, f, indent=2, ensure_ascii=False)

print("\nTAHAP 28E SELESAI")
print("Laporan:", OUTPUT)
print("Status: Belum siap training lima kelas.")
