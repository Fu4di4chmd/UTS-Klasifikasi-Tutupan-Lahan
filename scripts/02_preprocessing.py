
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
RASTER_DIR = BASE / "data" / "sentinel2"
OUTPUT = BASE / "outputs" / "reports"

OUTPUT.mkdir(parents=True, exist_ok=True)

BANDS = [
    "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B11", "B12"
]

FEATURES = BANDS + ["NDVI", "NDMI", "MNDWI", "NDBI"]

# =====================================================
# 2. BACA GEOJSON
# =====================================================

geojson_files = list(GEOJSON_DIR.glob("*.geojson"))
assert len(geojson_files) == 1, "Harus ada satu GeoJSON."

gdf = gpd.read_file(geojson_files[0])

gdf["class"] = (
    gdf["class"].astype(str).str.strip().str.lower()
)

assert gdf.crs is not None, "CRS GeoJSON tidak tersedia."

gdf["polygon_id"] = np.arange(1, len(gdf) + 1)

# =====================================================
# 3. TEMUKAN 10 BAND
# =====================================================

tiff_files = list(RASTER_DIR.rglob("*.tif"))
tiff_files += list(RASTER_DIR.rglob("*.tiff"))

band_paths = {}

for band in BANDS:
    matches = [
        p for p in tiff_files
        if f"_{band}_" in p.name.upper()
    ]
    assert len(matches) == 1, (
        f"Band {band} tidak ditemukan atau duplikat."
    )
    band_paths[band] = matches[0]

# =====================================================
# 4. BACA RASTER DAN EKSTRAK PIXEL
# =====================================================

with rasterio.open(band_paths["B04"]) as ref:
    crs = ref.crs
    transform = ref.transform
    shape = (ref.height, ref.width)

gdf = gdf.to_crs(crs)

# Raster ID poligon
# Piksel yang tidak berada di poligon bernilai 0
polygon_raster = rasterize(
    [
        (geom, int(pid))
        for geom, pid in zip(gdf.geometry, gdf["polygon_id"])
    ],
    out_shape=shape,
    transform=transform,
    fill=0,
    dtype="int32"
)

# Hanya baca piksel di dalam poligon
rows, cols = np.where(polygon_raster > 0)
polygon_ids = polygon_raster[rows, cols]

print("Piksel di dalam poligon:", len(rows))

data = {}

for band in BANDS:
    with rasterio.open(band_paths[band]) as src:
        assert src.crs == crs
        assert src.transform == transform
        assert (src.height, src.width) == shape

        array = src.read(1)
        nilai = array[rows, cols].astype(np.float32)

        if src.nodata is not None:
            nilai[array[rows, cols] == src.nodata] = np.nan

        data[band] = nilai

        del array

df = pd.DataFrame(data)

df["polygon_id"] = polygon_ids

label_lookup = gdf.set_index("polygon_id")["class"]
df["label"] = df["polygon_id"].map(label_lookup)

# =====================================================
# 5. FILTER PIKSEL TIDAK VALID
# =====================================================

mask = np.isfinite(df[BANDS]).all(axis=1)
mask &= (df[BANDS] > 0).all(axis=1)

df = df.loc[mask].copy()

# =====================================================
# 6. HITUNG INDEKS SPEKTRAL
# =====================================================

def indeks(a, b):
    penyebut = a + b
    return np.divide(
        a - b,
        penyebut,
        out=np.zeros_like(a, dtype=np.float32),
        where=penyebut != 0
    )

b03 = df["B03"].to_numpy(dtype=np.float32)
b04 = df["B04"].to_numpy(dtype=np.float32)
b08 = df["B08"].to_numpy(dtype=np.float32)
b11 = df["B11"].to_numpy(dtype=np.float32)

df["NDVI"] = indeks(b08, b04)
df["NDMI"] = indeks(b08, b11)
df["MNDWI"] = indeks(b03, b11)
df["NDBI"] = indeks(b11, b08)

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=FEATURES)

# =====================================================
# 7. PEMBAGIAN TRAINING / TESTING PER POLIGON
# =====================================================

# Poligon tanpa piksel valid tidak ikut pembagian
poligon_aktif = (
    gdf[gdf["polygon_id"].isin(df["polygon_id"].unique())]
    [["polygon_id", "class"]]
    .copy()
)

train_ids, test_ids = train_test_split(
    poligon_aktif["polygon_id"].to_numpy(),
    test_size=0.2,
    random_state=42,
    stratify=poligon_aktif["class"].to_numpy()
)

train_df = df[df["polygon_id"].isin(train_ids)].copy()
test_df = df[df["polygon_id"].isin(test_ids)].copy()

assert set(train_ids).isdisjoint(set(test_ids))

# =====================================================
# 8. SIMPAN HASIL
# =====================================================

train_df.to_csv(OUTPUT / "training.csv", index=False)
test_df.to_csv(OUTPUT / "testing.csv", index=False)

print("\n" + "=" * 55)
print("✅ PREPROCESSING SELESAI")
print("=" * 55)

print("Total sampel valid:", len(df))
print("Jumlah fitur:", len(FEATURES))
print("Poligon aktif:", len(poligon_aktif))

print("\nData training:")
print("Piksel:", len(train_df))
print("Poligon:", len(train_ids))

print("\nData testing:")
print("Piksel:", len(test_df))
print("Poligon:", len(test_ids))

print("\nDistribusi sampel:")
print(df["label"].value_counts())

print("\nTidak ada kebocoran poligon:", True)
print("File disimpan di:", OUTPUT)
