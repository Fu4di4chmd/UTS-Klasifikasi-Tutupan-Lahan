"""
KOREKSI FINAL WATER BODY
- Menghilangkan water body yang nyasar di persawahan
- Menghilangkan water body tipis di pinggir pulau / garis pantai
- Piksel yang dikoreksi langsung diganti ke kelas sekitar (bukan NoData)

Jalankan:
py -3.9 scripts/20_koreksi_waterbody_final.py
"""

from pathlib import Path
import json
import numpy as np
import rasterio
from scipy.ndimage import convolve, label
from rasterio.enums import Resampling
from PIL import Image

# ============================================================
# 1. PATH
# ============================================================

BASE = Path(__file__).resolve().parent.parent

# Pakai hasil screening Dynamic World sebagai input awal
INPUT_RF = (
    BASE / "outputs" / "maps" / "klasifikasi_rf19_waterbody_screening_100m.tif"
)

# Referensi air dari Dynamic World
INPUT_DW = (
    BASE / "data" / "prediksi" / "referensi_water_jatim_100m_2026.tif"
)

OUTPUT_DIR = BASE / "outputs" / "maps"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_TIF = OUTPUT_DIR / "klasifikasi_rf19_waterbody_final_100m.tif"
OUTPUT_PNG = OUTPUT_DIR / "klasifikasi_rf19_waterbody_final_100m.png"
OUTPUT_JSON = OUTPUT_DIR / "statistik_koreksi_waterbody_final.json"

# ============================================================
# 2. PARAMETER
# ============================================================

# Kode kelas
K_BUILD = 1
K_FARM = 2
K_WATER = 3
K_FOREST = 4
K_MANGROVE = 5
K_NODATA = 0

# Aturan koreksi
DW_KEEP_STRONG = 70     # kalau DW >= 70, water body dianggap kuat -> dipertahankan
DW_REMOVE_WEAK = 40     # kalau DW < 40, water body dianggap lemah -> kandidat koreksi
MIN_CLUSTER_SIZE = 6    # patch water body kecil < 6 piksel akan dibersihkan
FARM_NEIGHBOR_MIN = 12  # kalau sekitar piksel didominasi sawah pada jendela 5x5
COAST_BUFFER_SIZE = 5   # 5x5 = buffer 2 piksel dari NoData/tepi pulau

WARNA = {
    0: (0, 0, 0, 0),
    1: (128, 128, 128, 255),   # built up
    2: (255, 215, 0, 255),     # farm land
    3: (33, 150, 243, 255),    # water body
    4: (34, 139, 34, 255),     # forest
    5: (0, 206, 209, 255),     # mangrove
}

# ============================================================
# 3. FUNGSI BANTU
# ============================================================

def save_preview_png(raster_path, png_path):
    with rasterio.open(raster_path) as src:
        max_size = 2200
        faktor = max(src.width / max_size, src.height / max_size, 1)
        new_width = max(1, int(src.width / faktor))
        new_height = max(1, int(src.height / faktor))

        kelas_png = src.read(
            1,
            out_shape=(new_height, new_width),
            resampling=Resampling.nearest
        )

    rgba = np.zeros((new_height, new_width, 4), dtype=np.uint8)
    for kode, warna in WARNA.items():
        rgba[kelas_png == kode] = warna

    Image.fromarray(rgba).save(png_path)


# ============================================================
# 4. VALIDASI INPUT
# ============================================================

print("=" * 65)
print("KOREKSI FINAL WATER BODY")
print("=" * 65)

if not INPUT_RF.exists():
    raise FileNotFoundError(f"Raster RF tidak ditemukan:\n{INPUT_RF}")

if not INPUT_DW.exists():
    raise FileNotFoundError(f"Raster Dynamic World tidak ditemukan:\n{INPUT_DW}")

# ============================================================
# 5. BACA DATA
# ============================================================

with rasterio.open(INPUT_RF) as rf_src, rasterio.open(INPUT_DW) as dw_src:
    print("Raster RF :", INPUT_RF.name)
    print("Raster DW :", INPUT_DW.name)
    print("Resolusi RF:", rf_src.res)
    print("Resolusi DW:", dw_src.res)

    if rf_src.shape != dw_src.shape:
        raise ValueError("Ukuran raster RF dan DW harus sama.")
    if rf_src.crs != dw_src.crs:
        raise ValueError("CRS raster RF dan DW harus sama.")
    if rf_src.transform != dw_src.transform:
        raise ValueError("Transform raster RF dan DW harus sama.")

    rf = rf_src.read(1)
    dw = dw_src.read(1).astype(np.float32)
    profile = rf_src.profile.copy()
    try:
        colormap = rf_src.colormap(1)
    except ValueError:
        colormap = None

# ============================================================
# 6. MASK DASAR
# ============================================================

water = rf == K_WATER
farm = rf == K_FARM
nodata = rf == K_NODATA

water_awal = int(water.sum())

# ============================================================
# 7. DETEKSI WATER BODY YANG MAU DIBERSIHKAN
# ============================================================

# A. Water body lemah menurut Dynamic World
weak_dw = water & (dw < DW_REMOVE_WEAK)

# B. Water body sangat kuat -> jangan disentuh
strong_dw = water & (dw >= DW_KEEP_STRONG)

# C. Water body di dekat pinggir pulau / tepi NoData
kernel_coast = np.ones((COAST_BUFFER_SIZE, COAST_BUFFER_SIZE), dtype=np.uint8)
near_nodata = convolve(nodata.astype(np.uint8), kernel_coast, mode="constant", cval=1) > 0
coastal_water = water & near_nodata

# D. Water body yang dikelilingi sawah
kernel_5 = np.ones((5, 5), dtype=np.uint8)
farm_neighbors = convolve(farm.astype(np.uint8), kernel_5, mode="constant", cval=0)
farm_dominated = water & (farm_neighbors >= FARM_NEIGHBOR_MIN)

# E. Patch water body kecil
labels, n_groups = label(water, structure=np.ones((3, 3), dtype=np.uint8))
sizes = np.bincount(labels.ravel())
small_cluster = np.zeros_like(water, dtype=bool)
if len(sizes) > 1:
    small_cluster = (sizes[labels] < MIN_CLUSTER_SIZE) & water
    small_cluster[labels == 0] = False

# Gabungkan kandidat koreksi
candidate = (weak_dw | coastal_water | farm_dominated | small_cluster) & water

# Lindungi water body yang sangat kuat menurut DW
candidate &= ~strong_dw

# ============================================================
# 8. TENTUKAN KELAS PENGGANTI
# ============================================================

# Pengganti diambil dari kelas non-water di sekitar piksel
kernel_replace = np.ones((5, 5), dtype=np.uint8)

score_build = convolve((rf == K_BUILD).astype(np.uint8), kernel_replace, mode="constant", cval=0)
score_farm = convolve((rf == K_FARM).astype(np.uint8), kernel_replace, mode="constant", cval=0)
score_forest = convolve((rf == K_FOREST).astype(np.uint8), kernel_replace, mode="constant", cval=0)
score_mangrove = convolve((rf == K_MANGROVE).astype(np.uint8), kernel_replace, mode="constant", cval=0)

scores = np.stack([score_build, score_farm, score_forest, score_mangrove], axis=0)
kelas_opsi = np.array([K_BUILD, K_FARM, K_FOREST, K_MANGROVE], dtype=np.uint8)

idx_max = np.argmax(scores, axis=0)
replacement = kelas_opsi[idx_max]

# Jika semua tetangga 0, fallback ke farm land
all_zero = scores.max(axis=0) == 0
replacement[all_zero] = K_FARM

# ============================================================
# 9. TERAPKAN KOREKSI
# ============================================================

hasil = rf.copy()
hasil[candidate] = replacement[candidate]

water_akhir = int((hasil == K_WATER).sum())
piksel_koreksi = int(candidate.sum())

# Statistik alasan
jml_weak_dw = int(weak_dw.sum())
jml_coastal = int(coastal_water.sum())
jml_farmdom = int(farm_dominated.sum())
jml_small = int(small_cluster.sum())

# Statistik hasil pengganti
ganti_build = int(((candidate) & (replacement == K_BUILD)).sum())
ganti_farm = int(((candidate) & (replacement == K_FARM)).sum())
ganti_forest = int(((candidate) & (replacement == K_FOREST)).sum())
ganti_mangrove = int(((candidate) & (replacement == K_MANGROVE)).sum())

# ============================================================
# 10. SIMPAN OUTPUT
# ============================================================

profile.update(
    dtype="uint8",
    count=1,
    nodata=0,
    compress="deflate",
    tiled=True,
    blockxsize=256,
    blockysize=256,
)

with rasterio.open(OUTPUT_TIF, "w", **profile) as dst:
    dst.write(hasil, 1)
    if colormap is not None:
        dst.write_colormap(1, colormap)
    else:
        dst.write_colormap(1, WARNA)

    dst.set_band_description(
        1,
        "RF 19 fitur - water body final correction"
    )

save_preview_png(OUTPUT_TIF, OUTPUT_PNG)

laporan = {
    "input_rf": INPUT_RF.name,
    "input_dw": INPUT_DW.name,
    "water_body_awal": water_awal,
    "water_body_akhir": water_akhir,
    "piksel_dikoreksi": piksel_koreksi,
    "jumlah_kelompok_water_body": int(n_groups),
    "alasan_kandidat": {
        "weak_dynamic_world_lt_40": jml_weak_dw,
        "near_coast_or_nodata": jml_coastal,
        "farm_dominated_5x5": jml_farmdom,
        "small_cluster_lt_6": jml_small,
    },
    "kelas_pengganti": {
        "build_up": ganti_build,
        "farm_land": ganti_farm,
        "forest": ganti_forest,
        "mangrove": ganti_mangrove,
    },
    "catatan": [
        "Koreksi ini adalah post-processing, bukan training ulang.",
        "Water body kuat (DW >= 70) dipertahankan.",
        "Water body lemah, di tepi pulau, atau di area sawah dibersihkan."
    ]
}
OUTPUT_JSON.write_text(json.dumps(laporan, indent=2, ensure_ascii=False), encoding="utf-8")

# ============================================================
# 11. TAMPILKAN HASIL
# ============================================================

print("\n" + "=" * 65)
print("KOREKSI SELESAI")
print("=" * 65)
print(f"Water body awal   : {water_awal:,} piksel")
print(f"Water body akhir  : {water_akhir:,} piksel")
print(f"Piksel dikoreksi  : {piksel_koreksi:,} piksel")

print("\nAlasan kandidat:")
print(f"- DW < {DW_REMOVE_WEAK}                : {jml_weak_dw:,}")
print(f"- Dekat tepi pulau / NoData   : {jml_coastal:,}")
print(f"- Dikelilingi sawah           : {jml_farmdom:,}")
print(f"- Patch kecil                 : {jml_small:,}")

print("\nDiganti menjadi:")
print(f"- Built-up  : {ganti_build:,}")
print(f"- Farm land : {ganti_farm:,}")
print(f"- Forest    : {ganti_forest:,}")
print(f"- Mangrove  : {ganti_mangrove:,}")

print("\nOutput raster :", OUTPUT_TIF)
print("Preview PNG   :", OUTPUT_PNG)
print("Laporan JSON  :", OUTPUT_JSON)
print("\nModel Random Forest tidak diubah.")