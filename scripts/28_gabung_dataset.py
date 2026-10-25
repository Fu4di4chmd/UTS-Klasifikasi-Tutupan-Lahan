
from pathlib import Path
import pandas as pd
import numpy as np
import json

BASE = Path(__file__).resolve().parents[1]
FOLDER = BASE / "outputs" / "datasets"
FOLDER.mkdir(parents=True, exist_ok=True)

BLOK = [
    "B_32_459",
    "B_34_459",
    "B_34_460",
    "B_28_459",
    "B_28_456"
]

FITUR = [
    "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B11", "B12",
    "NDVI", "NDMI", "MNDWI", "NDBI",
    "NDWI", "MVI", "REMI",
    "MANGROVE_8A_11", "VEG_8A_4"
]

KELAS = [
    "build up",
    "farm land",
    "water body",
    "forest",
    "mangrove"
]

print("=" * 65)
print("TAHAP 28 - PENGGABUNGAN DAN AUDIT DATASET")
print("=" * 65)

semua = []

for blok in BLOK:
    path = FOLDER / f"tahap28d_sampel_{blok}.csv"

    if not path.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig")

    wajib = ["id_poligon", "kelas", "x_utm", "y_utm"] + FITUR
    kurang = [c for c in wajib if c not in df.columns]

    if kurang:
        raise ValueError(
            f"Kolom tidak ditemukan pada {blok}: {kurang}"
        )

    df["blok_id"] = blok
    df["kelas"] = df["kelas"].astype(str).str.strip().str.lower()

    semua.append(df)
    print(f"{blok}: {len(df):,} sampel")

data = pd.concat(semua, ignore_index=True)
jumlah_awal = len(data)

print("\nMemeriksa nilai numerik dan fitur...")

for kolom in FITUR + ["x_utm", "y_utm"]:
    data[kolom] = pd.to_numeric(data[kolom], errors="coerce")

data[FITUR] = data[FITUR].replace(
    [np.inf, -np.inf], np.nan
)

tidak_valid = data[FITUR + ["x_utm", "y_utm"]].isna().any(axis=1)
jumlah_tidak_valid = int(tidak_valid.sum())

data = data.loc[~tidak_valid].copy()

kelas_tidak_dikenal = sorted(set(data["kelas"]) - set(KELAS))

if kelas_tidak_dikenal:
    raise ValueError(
        f"Ditemukan kelas tidak dikenal: {kelas_tidak_dikenal}"
    )

print("Memeriksa duplikasi posisi piksel...")

# Koordinat dibulatkan sampai 0,01 meter untuk
# mendeteksi posisi yang identik secara praktis.
data["_x"] = data["x_utm"].round(2)
data["_y"] = data["y_utm"].round(2)

konflik = (
    data.groupby(["_x", "_y"])["kelas"]
    .nunique()
)

jumlah_konflik = int((konflik > 1).sum())

if jumlah_konflik > 0:
    print(
        f"PERINGATAN: {jumlah_konflik} posisi memiliki "
        "lebih dari satu label kelas."
    )

# Untuk posisi identik, pertahankan satu sampel.
data = data.drop_duplicates(
    subset=["_x", "_y"],
    keep="first"
).copy()

jumlah_duplikat = jumlah_awal - jumlah_tidak_valid - len(data)

data.drop(columns=["_x", "_y"], inplace=True)

print("\nMenghitung distribusi setiap kelas...")

ringkasan = []

for kelas in KELAS:
    subset = data[data["kelas"] == kelas]

    ringkasan.append({
        "kelas": kelas,
        "jumlah_piksel": int(len(subset)),
        "jumlah_poligon_unik": int(
            subset["id_poligon"].nunique()
        ),
        "jumlah_blok": int(
            subset["blok_id"].nunique()
        )
    })

statistik = pd.DataFrame(ringkasan)

output_dataset = FOLDER / "dataset_final_20m_19fitur.csv"
output_statistik = FOLDER / "statistik_dataset_final.csv"
output_laporan = FOLDER / "laporan_dataset_final.json"

data.to_csv(output_dataset, index=False)
statistik.to_csv(output_statistik, index=False)

laporan = {
    "jumlah_blok": len(BLOK),
    "daftar_blok": BLOK,
    "jumlah_fitur": len(FITUR),
    "jumlah_sampel_awal": jumlah_awal,
    "jumlah_sampel_tidak_valid": jumlah_tidak_valid,
    "jumlah_duplikat_posisi_identik": jumlah_duplikat,
    "jumlah_posisi_konflik_label": jumlah_konflik,
    "jumlah_sampel_akhir": len(data),
    "jumlah_poligon_unik_total": int(
        data["id_poligon"].nunique()
    ),
    "distribusi": ringkasan
}

with open(output_laporan, "w", encoding="utf-8") as f:
    json.dump(laporan, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 65)
print("HASIL AUDIT DATASET")
print("=" * 65)

print(f"Sampel awal              : {jumlah_awal:,}")
print(f"Sampel tidak valid       : {jumlah_tidak_valid:,}")
print(f"Duplikat posisi identik  : {jumlah_duplikat:,}")
print(f"Konflik label posisi     : {jumlah_konflik:,}")
print(f"Sampel setelah audit     : {len(data):,}")
print(f"Jumlah fitur             : {len(FITUR)}")
print(f"Poligon unik             : {data['id_poligon'].nunique()}")

print("\nDISTRIBUSI KELAS")
print(statistik.to_string(index=False))

print("\nOUTPUT")
print("Dataset  :", output_dataset)
print("Statistik:", output_statistik)
print("Laporan  :", output_laporan)

if jumlah_konflik:
    print("\nPERHATIAN: Konflik label perlu diperiksa sebelum training.")

if (statistik["jumlah_poligon_unik"] < 2).any():
    print("\nPERHATIAN: Ada kelas dengan kurang dari 2 poligon.")

print("\nTAHAP 28 - PENGGABUNGAN SELESAI")
print("Training Random Forest belum dilakukan.")
