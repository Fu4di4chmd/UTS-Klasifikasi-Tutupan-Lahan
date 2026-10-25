"""Preprocessing langsung dari GeoJSON terbaru + Sentinel-2 14 band menjadi 19 fitur (Penambahan fitur Mangrove 8A, 11, 4).

Jalankan dari root proyek:
    py -3.9 scripts/02_preprocessing.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from sklearn.model_selection import train_test_split

BASE = Path(__file__).resolve().parent.parent
GEOJSON_DIR = BASE / "data" / "geojson"
RASTER_PATH = BASE / "data" / "prediksi" / "sentinel2_jatim_training_buffer3km_14fitur_100m.tif"
OUTPUT = BASE / "outputs" / "reports" / "eksperimen_mangrove_17fitur"

BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
FEATURES_14 = BANDS + ["NDVI", "NDMI", "MNDWI", "NDBI"]
# Menambahkan 2 fitur baru khusus untuk membedakan Mangrove vs Air vs Hutan Biasa
FEATURES_EKSPERIMEN = FEATURES_14 + ["NDWI", "MVI", "REMI", "MANGROVE_8A_11", "VEG_8A_4"]
CLASSES = ["build up", "farm land", "water body", "forest", "mangrove"]
EPS = 1e-6


def safe_ratio(numerator, denominator, absolute=False):
    n = np.asarray(numerator, dtype=np.float64)
    d = np.asarray(denominator, dtype=np.float64)
    if absolute:
        n, d = np.abs(n), np.abs(d)
    valid = np.isfinite(n) & np.isfinite(d) & (np.abs(d) > EPS)
    result = np.full(n.shape, np.nan, dtype=np.float64)
    np.divide(n, d, out=result, where=valid)
    return result.astype(np.float32)


def main():
    print("=" * 65)
    print("PREPROCESSING GEOJSON TERBARU -> RANDOM FOREST FITUR MANGROVE")
    print("=" * 65)

    files = sorted(GEOJSON_DIR.glob("*.geojson"))
    if len(files) != 1:
        raise ValueError(f"Harus tepat satu GeoJSON pada {GEOJSON_DIR}. Ditemukan: {[f.name for f in files]}")
    if not RASTER_PATH.is_file():
        raise FileNotFoundError(f"Raster tidak ditemukan: {RASTER_PATH}")

    gdf = gpd.read_file(files[0])
    if gdf.crs is None:
        raise ValueError("GeoJSON tidak mempunyai CRS.")
    if "class" not in gdf.columns:
        raise ValueError("Kolom 'class' tidak ditemukan di GeoJSON.")
    gdf["class"] = gdf["class"].astype(str).str.strip().str.lower()
    if not gdf["class"].isin(CLASSES).all():
        raise ValueError(f"Label tidak dikenal: {gdf.loc[~gdf['class'].isin(CLASSES), 'class'].unique().tolist()}")
    if gdf.geometry.isna().any() or gdf.geometry.is_empty.any() or not gdf.geometry.is_valid.all():
        raise ValueError("Ada geometri poligon kosong atau tidak valid. Perbaiki GeoJSON terlebih dahulu.")
    if not gdf.geom_type.isin(["Polygon", "MultiPolygon"]).all():
        raise ValueError("Semua geometri harus Polygon/MultiPolygon.")

    gdf["polygon_id"] = np.arange(1, len(gdf) + 1, dtype=np.int32)
    print("GeoJSON:", files[0].name)
    print("Jumlah poligon GeoJSON:", len(gdf))
    print("Distribusi poligon:\n", gdf["class"].value_counts().to_string())

    with rasterio.open(RASTER_PATH) as src:
        if src.count != len(FEATURES_14) or list(src.descriptions) != FEATURES_14:
            raise ValueError(
                f"Raster harus berisi 14 band dengan urutan {FEATURES_14}.\n"
                f"Terbaca: {list(src.descriptions)}"
            )
        if src.crs is None:
            raise ValueError("Raster tidak mempunyai CRS.")
        gdf = gdf.to_crs(src.crs)
        print("Raster:", RASTER_PATH.name, "| ukuran:", src.width, "x", src.height)

        polygon_raster = rasterize(
            [(geom, int(pid)) for geom, pid in zip(gdf.geometry, gdf["polygon_id"])],
            out_shape=(src.height, src.width), transform=src.transform,
            fill=0, dtype="int32"
        )
        rows, cols = np.where(polygon_raster > 0)
        if len(rows) == 0:
            raise ValueError("Tidak ada piksel dari GeoJSON yang mengenai raster.")
        ids = polygon_raster[rows, cols]
        print("Piksel yang tercakup poligon:", len(rows))

        data = {}
        for i, name in enumerate(FEATURES_14, start=1):
            band = src.read(i, masked=True)
            values = np.ma.asarray(band[rows, cols], dtype=np.float32).filled(np.nan)
            data[name] = np.asarray(values, dtype=np.float32)
            del band
            print(f"Ekstraksi {i:02d}/14: {name}")

    df = pd.DataFrame(data)
    df["polygon_id"] = ids
    df["label"] = df["polygon_id"].map(gdf.set_index("polygon_id")["class"])
    valid14 = np.isfinite(df[FEATURES_14].to_numpy(dtype=np.float64)).all(axis=1)
    valid14 &= (df[BANDS].to_numpy(dtype=np.float64) > 0).all(axis=1)
    df = df.loc[valid14].copy()
    print("Piksel valid 14 fitur:", len(df))

    # --- PERHITUNGAN INDEKS ---
    df["NDWI"] = safe_ratio(df["B03"] - df["B08"], df["B03"] + df["B08"])
    df["MVI"] = safe_ratio(df["B08"] - df["B03"], df["B11"] - df["B03"], absolute=True)
    df["REMI"] = safe_ratio(df["B05"] - df["B04"], df["B11"] - df["B03"])
    
    # INDEKS KHUSUS MANGROVE (Sesuai Syarat UTS: Band 8A, 11, 4)
    df["MANGROVE_8A_11"] = safe_ratio(df["B8A"] - df["B11"], df["B8A"] + df["B11"])
    df["VEG_8A_4"] = safe_ratio(df["B8A"] - df["B04"], df["B8A"] + df["B04"])

    for feature in ["NDWI", "MVI", "REMI", "MANGROVE_8A_11", "VEG_8A_4"]:
        print(f"{feature} tidak valid: {int((~np.isfinite(df[feature])).sum())} piksel")
        
    valid_all = np.isfinite(df[FEATURES_EKSPERIMEN].to_numpy(dtype=np.float64)).all(axis=1)
    df = df.loc[valid_all].copy()
    if df.empty:
        raise ValueError("Tidak ada piksel valid.")

    active = gdf.loc[gdf["polygon_id"].isin(df["polygon_id"].unique()), ["polygon_id", "class"]].copy()
    counts = active["class"].value_counts()
    if len(counts) != len(CLASSES) or (counts < 2).any():
        raise ValueError(f"Tidak semua kelas memiliki >=2 poligon valid:\n{counts}")
    
    train_ids, test_ids = train_test_split(
        active["polygon_id"].to_numpy(), test_size=0.2,
        random_state=42, stratify=active["class"].to_numpy()
    )
    train = df.loc[df["polygon_id"].isin(train_ids)].copy()
    test = df.loc[df["polygon_id"].isin(test_ids)].copy()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    columns = FEATURES_EKSPERIMEN + ["polygon_id", "label"]
    
    # Menyimpan dengan nama yang sama agar script selanjutnya tidak error
    train[columns].to_csv(OUTPUT / "training_17fitur.csv", index=False)
    test[columns].to_csv(OUTPUT / "testing_17fitur.csv", index=False)
    
    summary = pd.DataFrame([
        {"split": "training", "pixels": len(train), "polygons": len(train_ids)},
        {"split": "testing", "pixels": len(test), "polygons": len(test_ids)}
    ])
    summary.to_csv(OUTPUT / "ringkasan.csv", index=False)
    (OUTPUT / "fitur_model.txt").write_text("\n".join(FEATURES_EKSPERIMEN) + "\n", encoding="utf-8")

    print("\n" + "=" * 65)
    print("PREPROCESSING BERHASIL DENGAN FITUR MANGROVE BARU")
    print("Total piksel valid:", len(df))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()