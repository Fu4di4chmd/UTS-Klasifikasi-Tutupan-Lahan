
# ==========================================================
# TAHAP 1 — DATA UNDERSTANDING
# KLASIFIKASI TUTUPAN LAHAN JAWA TIMUR
# ==========================================================

from pathlib import Path
import geopandas as gpd
import rasterio
import pandas as pd

# ----------------------------------------------------------
# 1. LOKASI FOLDER PROYEK
# ----------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

GEOJSON_DIR = BASE_DIR / "data" / "geojson"
SENTINEL_DIR = BASE_DIR / "data" / "sentinel2"
OUTPUT_DIR = BASE_DIR / "outputs" / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------
# 2. MEMBACA DATA GEOJSON
# ----------------------------------------------------------

geojson_files = sorted(GEOJSON_DIR.glob("*.geojson"))

if len(geojson_files) != 1:
    raise FileNotFoundError(
        "Folder data/geojson harus berisi tepat satu "
        "file GeoJSON terbaru."
    )

geojson_path = geojson_files[0]

gdf = gpd.read_file(geojson_path)

print("=" * 60)
print("DATA UNDERSTANDING — GEOJSON")
print("=" * 60)

print("Nama file:", geojson_path.name)
print("Jumlah poligon:", len(gdf))
print("Sistem koordinat:", gdf.crs)
print("Kolom:", list(gdf.columns))

if "class" not in gdf.columns:
    raise ValueError(
        "GeoJSON tidak memiliki kolom 'class'."
    )

# Rapikan penulisan nama kelas
gdf["class"] = (
    gdf["class"]
    .astype("string")
    .str.strip()
    .str.lower()
)

print("\nDistribusi kelas:")
print(gdf["class"].value_counts(dropna=False))

# ----------------------------------------------------------
# 3. VALIDASI GEOJSON
# ----------------------------------------------------------

kelas_valid = [
    "build up",
    "farm land",
    "water body",
    "forest",
    "mangrove"
]

kelas_tidak_dikenal = gdf.loc[
    ~gdf["class"].isin(kelas_valid),
    "class"
].unique()

print("\nPemeriksaan kualitas GeoJSON:")
print("Geometri kosong:", gdf.geometry.isna().sum())
print("Geometri tidak valid:", (~gdf.geometry.is_valid).sum())
print("Label tidak dikenal:", list(kelas_tidak_dikenal))

if gdf.crs is None:
    raise ValueError(
        "GeoJSON tidak memiliki informasi CRS."
    )

if len(gdf) != 302:
    print(
        "PERHATIAN: Jumlah poligon berbeda dari "
        "302 poligon yang direncanakan."
    )

if (
    gdf.geometry.isna().any()
    or (~gdf.geometry.is_valid).any()
    or len(kelas_tidak_dikenal) > 0
):
    raise ValueError(
        "Terdapat masalah pada GeoJSON. "
        "Periksa data sebelum preprocessing."
    )

# ----------------------------------------------------------
# 4. MEMBACA FILE TIFF SENTINEL-2
# ----------------------------------------------------------

print("\n" + "=" * 60)
print("DATA UNDERSTANDING — SENTINEL-2")
print("=" * 60)

tiff_files = sorted(
    list(SENTINEL_DIR.rglob("*.tif"))
    + list(SENTINEL_DIR.rglob("*.tiff"))
)

if not tiff_files:
    raise FileNotFoundError(
        "Tidak ada TIFF pada folder data/sentinel2."
    )

bands = [
    "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B11", "B12"
]

band_paths = {}

for band in bands:
    cocok = [
        path for path in tiff_files
        if f"_{band}_" in path.name.upper()
    ]

    if len(cocok) != 1:
        raise ValueError(
            f"Band {band} harus memiliki tepat satu "
            f"file TIFF. Ditemukan: {len(cocok)}"
        )

    band_paths[band] = cocok[0]

print("Jumlah band klasifikasi:", len(band_paths))

info_bands = []

for band, path in band_paths.items():
    with rasterio.open(path) as src:
        info_bands.append({
            "Band": band,
            "File": path.name,
            "Width": src.width,
            "Height": src.height,
            "CRS": str(src.crs),
            "Resolusi_X": src.res[0],
            "Resolusi_Y": src.res[1]
        })

df_bands = pd.DataFrame(info_bands)

print("\nInformasi band Sentinel-2:")
print(
    df_bands[
        ["Band", "Width", "Height", "CRS", "Resolusi_X"]
    ].to_string(index=False)
)

# ----------------------------------------------------------
# 5. MEMERIKSA KESESUAIAN ANTARBAND
# ----------------------------------------------------------

with rasterio.open(band_paths["B04"]) as referensi:
    crs_ref = referensi.crs
    transform_ref = referensi.transform
    ukuran_ref = (referensi.height, referensi.width)

for band, path in band_paths.items():
    with rasterio.open(path) as src:
        assert src.crs == crs_ref, (
            f"CRS band {band} berbeda."
        )
        assert src.transform == transform_ref, (
            f"Transformasi band {band} berbeda."
        )
        assert (src.height, src.width) == ukuran_ref, (
            f"Ukuran band {band} berbeda."
        )

print("\n✅ Sepuluh band memiliki grid yang sama.")

# ----------------------------------------------------------
# 6. MEMERIKSA TIFF TRUE COLOR
# ----------------------------------------------------------

true_color_files = [
    p for p in tiff_files
    if "true_color" in p.name.lower()
]

if len(true_color_files) != 1:
    raise ValueError(
        "Harus tersedia tepat satu TIFF True Color. "
        f"Ditemukan: {len(true_color_files)}"
    )

with rasterio.open(true_color_files[0]) as src:
    print("\nTrue Color:", true_color_files[0].name)
    print("Ukuran:", src.width, "x", src.height)
    print("Jumlah channel:", src.count)
    print("CRS:", src.crs)

# ----------------------------------------------------------
# 7. SIMPAN RINGKASAN DATA
# ----------------------------------------------------------

distribusi = (
    gdf["class"]
    .value_counts()
    .rename_axis("Kelas")
    .reset_index(name="Jumlah_Poligon")
)

distribusi.to_csv(
    OUTPUT_DIR / "distribusi_geojson.csv",
    index=False
)

df_bands.to_csv(
    OUTPUT_DIR / "informasi_band.csv",
    index=False
)

print("\n" + "=" * 60)
print("✅ DATA UNDERSTANDING SELESAI")
print("=" * 60)
print("Total poligon:", len(gdf))
print("Jumlah kelas:", gdf["class"].nunique())
print("Jumlah band:", len(band_paths))
print("Laporan tersimpan di:", OUTPUT_DIR)
