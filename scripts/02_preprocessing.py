
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

from rasterio.features import rasterize
from sklearn.model_selection import train_test_split

# =====================================================
# 1. LOKASI DATA
# =====================================================

BASE = Path(__file__).resolve().parent.parent

GEOJSON_DIR = BASE / "data" / "geojson"

RASTER_PATH = (
    BASE / "data" / "prediksi"
    / "sentinel2_jatim_training_buffer3km_14fitur_100m.tif"
)

OUTPUT = BASE / "outputs" / "reports"
OUTPUT.mkdir(parents=True, exist_ok=True)

BANDS = [
    "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B11", "B12"
]

FEATURES = BANDS + [
    "NDVI", "NDMI", "MNDWI", "NDBI"
]

CLASSES = [
    "build up",
    "farm land",
    "water body",
    "forest",
    "mangrove"
]

print("=" * 60)
print("PREPROCESSING SENTINEL-2 KOMPOSIT JAWA TIMUR")
print("=" * 60)

# =====================================================
# 2. BACA GEOJSON
# =====================================================

geojson_files = list(GEOJSON_DIR.glob("*.geojson"))

assert len(geojson_files) == 1, (
    "Folder geojson harus berisi tepat satu file GeoJSON."
)

gdf = gpd.read_file(geojson_files[0])

assert gdf.crs is not None, (
    "CRS GeoJSON tidak tersedia."
)

assert "class" in gdf.columns, (
    "Kolom class tidak ditemukan."
)

gdf["class"] = (
    gdf["class"].astype(str).str.strip().str.lower()
)

assert gdf["class"].isin(CLASSES).all(), (
    "Ada label kelas yang tidak sesuai."
)

assert gdf.geometry.notna().all(), (
    "Ada geometri kosong."
)

assert gdf.is_valid.all(), (
    "Ada geometri poligon tidak valid."
)

gdf["polygon_id"] = np.arange(1, len(gdf) + 1)

print("Jumlah poligon GeoJSON:", len(gdf))
print("Distribusi poligon:")
print(gdf["class"].value_counts())

# =====================================================
# 3. BACA RASTER KOMPOSIT
# =====================================================

assert RASTER_PATH.exists(), (
    f"Raster tidak ditemukan: {RASTER_PATH}"
)

with rasterio.open(RASTER_PATH) as src:

    assert src.crs is not None, (
        "CRS raster tidak tersedia."
    )

    assert src.count == len(FEATURES), (
        f"Raster harus mempunyai {len(FEATURES)} band."
    )

    raster_features = list(src.descriptions)

    assert raster_features == FEATURES, (
        "Nama atau urutan band raster tidak sesuai.\n"
        f"Raster: {raster_features}\n"
        f"Model : {FEATURES}"
    )

    print("\nRaster:", RASTER_PATH.name)
    print("Jumlah band:", src.count)
    print("Resolusi:", src.res)
    print("CRS:", src.crs)

    # Samakan CRS poligon dengan raster
    gdf = gdf.to_crs(src.crs)

    # =================================================
    # 4. RASTERISASI ID POLIGON
    # =================================================

    polygon_raster = rasterize(
        [
            (geom, int(pid))
            for geom, pid in zip(
                gdf.geometry,
                gdf["polygon_id"]
            )
        ],
        out_shape=(src.height, src.width),
        transform=src.transform,
        fill=0,
        dtype="int32"
    )

    rows, cols = np.where(polygon_raster > 0)

    polygon_ids = polygon_raster[rows, cols]

    print("\nPiksel di dalam poligon:", len(rows))

    if len(rows) == 0:
        raise ValueError(
            "Tidak ada piksel raster dalam poligon."
        )

    # =================================================
    # 5. EKSTRAK 14 FITUR
    # =================================================

    data = {}

    for i, feature in enumerate(FEATURES, start=1):

        # Baca satu band setiap iterasi agar hemat RAM
        array = src.read(i, masked=True)

        nilai = array[rows, cols].astype(
            np.float32
        )

        if np.ma.isMaskedArray(nilai):
            nilai = nilai.filled(np.nan)

        data[feature] = np.asarray(
            nilai, dtype=np.float32
        )

        print(
            f"Fitur {i:02d}/14 selesai: {feature}"
        )

        del array

# =====================================================
# 6. BUAT DATAFRAME
# =====================================================

df = pd.DataFrame(data)

df["polygon_id"] = polygon_ids

label_lookup = (
    gdf.set_index("polygon_id")["class"]
)

df["label"] = df["polygon_id"].map(
    label_lookup
)

# =====================================================
# 7. FILTER PIKSEL VALID
# =====================================================

mask = np.isfinite(df[FEATURES]).all(axis=1)

# Seluruh 10 band reflektansi harus > 0
mask &= (df[BANDS] > 0).all(axis=1)

df = df.loc[mask].copy()

if df.empty:
    raise ValueError(
        "Tidak ada sampel valid setelah filtering."
    )

# =====================================================
# 8. PEMBAGIAN TRAINING / TESTING
#    BERDASARKAN POLIGON
# =====================================================

poligon_aktif = (
    gdf[
        gdf["polygon_id"].isin(
            df["polygon_id"].unique()
        )
    ][["polygon_id", "class"]]
    .copy()
)

jumlah_per_kelas = (
    poligon_aktif["class"].value_counts()
)

assert (jumlah_per_kelas >= 2).all(), (
    "Setiap kelas harus memiliki minimal "
    "dua poligon aktif."
)

train_ids, test_ids = train_test_split(
    poligon_aktif["polygon_id"].to_numpy(),
    test_size=0.2,
    random_state=42,
    stratify=poligon_aktif["class"].to_numpy()
)

train_df = df[
    df["polygon_id"].isin(train_ids)
].copy()

test_df = df[
    df["polygon_id"].isin(test_ids)
].copy()

assert set(train_ids).isdisjoint(
    set(test_ids)
), "Terjadi kebocoran poligon."

# =====================================================
# 9. SIMPAN CSV
# =====================================================

train_df.to_csv(
    OUTPUT / "training.csv",
    index=False
)

test_df.to_csv(
    OUTPUT / "testing.csv",
    index=False
)

# =====================================================
# 10. LAPORAN
# =====================================================

print("\n" + "=" * 60)
print("PREPROCESSING KOMPOSIT SELESAI")
print("=" * 60)

print("Total sampel valid:", len(df))
print("Jumlah fitur:", len(FEATURES))
print("Poligon aktif:", len(poligon_aktif))

print("\nData training:")
print("Piksel:", len(train_df))
print("Poligon:", len(train_ids))
print(train_df["label"].value_counts())

print("\nData testing:")
print("Piksel:", len(test_df))
print("Poligon:", len(test_ids))
print(test_df["label"].value_counts())

print(
    "\nTidak ada kebocoran poligon:",
    set(train_ids).isdisjoint(set(test_ids))
)

print("\nFile tersimpan:")
print(OUTPUT / "training.csv")
print(OUTPUT / "testing.csv")
