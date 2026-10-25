
"""
TAHAP 27 - KOMPOSIT MULTI-TANGGAL SENTINEL-2 20 M
Tanpa Google Earth Engine.

Alur:
1. Cari citra Sentinel-2 L2A tahun 2026.
2. Pilih hingga 6 tanggal dengan cakupan area yang baik.
3. Mask awan dan bayangan dengan SCL.
4. Konversi DN menjadi reflektansi.
5. Hitung median per band dari beberapa tanggal.
6. Hitung 19 fitur dari median 10 band.
7. Simpan GeoTIFF dan laporan.

Area uji: Surabaya - Sidoarjo.

Jalankan:
py -3.9 scripts/27_komposit_multitanggal_20m.py
"""

from pathlib import Path
from datetime import date
import json
import warnings
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np
import rasterio
import planetary_computer

from affine import Affine
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds


# ============================================================
# 1. KONFIGURASI
# ============================================================

BASE = Path(__file__).resolve().parent.parent

OUT = BASE / "data/prediksi/sentinel2_20m_blok/B_28_456"
TEMP = OUT / "temp_tahap27"

OUT.mkdir(parents=True, exist_ok=True)
TEMP.mkdir(parents=True, exist_ok=True)

OUTPUT10 = OUT / "komposit_sentinel2_10band_20m.tif"
OUTPUT19 = OUT / "komposit_sentinel2_19fitur_20m.tif"
LAPORAN = OUT / "laporan_komposit_tahap27.json"

BBOX = [111.54417276745225, -7.960766406837997, 111.72586964979047, -7.779591070740225]

PERIODE = "2026-06-01/2026-09-30"

JUMLAH_TANGGAL = 6
MIN_TANGGAL = 3
JARAK_HARI = 5

MIN_VALID = 0.50
MAKS_KANDIDAT = 100

BANDS = [
    "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B11", "B12"
]

FEATURES = BANDS + [
    "NDVI", "NDMI", "MNDWI", "NDBI",
    "NDWI", "MVI", "REMI",
    "MANGROVE_8A_11", "VEG_8A_4"
]

CRS = "EPSG:32749"
RES = 20

# SCL 6 (water) sengaja dipertahankan.
SCL_BURUK = [0, 1, 3, 8, 9, 10, 11]


# ============================================================
# 2. FUNGSI
# ============================================================

def rasio(a, b, minimum=1e-6, absolut=False):
    if absolut:
        a = np.abs(a)
        b = np.abs(b)

    out = np.full(a.shape, np.nan, dtype=np.float32)

    valid = (
        np.isfinite(a)
        & np.isfinite(b)
        & (np.abs(b) >= minimum)
    )

    np.divide(a, b, out=out, where=valid)

    return out


def cek_metadata_offset(item):
    """Verifikasi offset reflektansi setiap produk."""
    if "product-metadata" not in item.assets:
        raise ValueError("Metadata produk tidak tersedia.")

    url = item.assets["product-metadata"].href

    with urllib.request.urlopen(url, timeout=90) as response:
        root = ET.fromstring(response.read())

    def ambil(tag):
        return [
            x.text.strip()
            for x in root.iter()
            if x.tag.split("}")[-1] == tag
            and x.text
            and x.text.strip()
        ]

    offsets = [
        float(x) for x in ambil("BOA_ADD_OFFSET")
    ]
    quants = [
        float(x) for x in ambil("BOA_QUANTIFICATION_VALUE")
    ]

    if not offsets or not quants:
        raise ValueError("Offset atau kuantifikasi tidak tersedia.")

    if not all(abs(x + 1000) < 0.01 for x in offsets):
        raise ValueError(f"Offset berbeda: {offsets}")

    if abs(quants[0] - 10000) > 0.01:
        raise ValueError(f"Quantification berbeda: {quants}")

    return True


def baca_band(url, transform, width, height, metode):
    with rasterio.open(url) as src:
        with WarpedVRT(
            src,
            crs=CRS,
            transform=transform,
            width=width,
            height=height,
            resampling=metode,
            nodata=0
        ) as vrt:
            return vrt.read(1)


def simpan_raster(path, data, transform, nama_band):
    profile = {
        "driver": "GTiff",
        "height": data.shape[1],
        "width": data.shape[2],
        "count": data.shape[0],
        "dtype": "float32",
        "crs": CRS,
        "transform": transform,
        "nodata": np.nan,
        "compress": "deflate",
        "predictor": 3,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "BIGTIFF": "IF_SAFER"
    }

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)

        for i, name in enumerate(nama_band, 1):
            dst.set_band_description(i, name)


# ============================================================
# 3. GRID TARGET 20 METER
# ============================================================

print("=" * 65)
print("TAHAP 27 - KOMPOSIT SENTINEL-2 MULTI-TANGGAL 20 M")
print("=" * 65)

left, bottom, right, top = transform_bounds(
    "EPSG:4326",
    CRS,
    *BBOX
)

left = np.floor(left / RES) * RES
bottom = np.floor(bottom / RES) * RES
right = np.ceil(right / RES) * RES
top = np.ceil(top / RES) * RES

width = int(round((right - left) / RES))
height = int(round((top - bottom) / RES))

transform = Affine(
    RES, 0, left,
    0, -RES, top
)

print("Grid:", width, "x", height)
print("Resolusi:", RES, "meter")
print("CRS:", CRS)


# ============================================================
# 4. PENCARIAN CITRA
# ============================================================

print("\nMencari citra Sentinel-2...")

catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace
)

items = list(catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=BBOX,
    datetime=PERIODE,
    query={"eo:cloud_cover": {"lt": 60}},
    max_items=MAKS_KANDIDAT
).items())

if not items:
    raise RuntimeError("Tidak ditemukan citra Sentinel-2.")

items.sort(
    key=lambda x: x.properties.get("eo:cloud_cover", 100)
)

print("Jumlah kandidat:", len(items))


# ============================================================
# 5. PILIH CITRA DAN SIMPAN DATA SEMENTARA
# ============================================================

dipilih = []
tanggal_terpilih = []

with rasterio.Env(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"
):

    for item in items:

        if len(dipilih) >= JUMLAH_TANGGAL:
            break

        if not all(
            b in item.assets
            for b in BANDS + ["SCL", "product-metadata"]
        ):
            continue

        tanggal = item.datetime.date()

        if any(
            abs((tanggal - t).days) < JARAK_HARI
            for t in tanggal_terpilih
        ):
            continue

        print("\n" + "-" * 55)
        print("Menguji:", item.id)
        print("Tanggal:", tanggal)
        print(
            "Awan metadata:",
            item.properties.get("eo:cloud_cover"),
            "%"
        )

        try:
            scl = baca_band(
                item.assets["SCL"].href,
                transform, width, height,
                Resampling.nearest
            )

            valid = ~np.isin(scl, SCL_BURUK)

            # Pastikan B11 punya cakupan data nyata.
            b11 = baca_band(
                item.assets["B11"].href,
                transform, width, height,
                Resampling.bilinear
            )

            valid &= np.isfinite(b11)
            valid &= b11 > 0

            persen = float(valid.mean())

            print(
                "Cakupan piksel valid:",
                round(persen * 100, 2), "%"
            )

            if persen < MIN_VALID:
                print("Dilewati: terlalu banyak piksel kosong.")
                continue

            cek_metadata_offset(item)

            print("Metadata offset: valid")

            scene = np.full(
                (10, height, width),
                np.nan,
                dtype=np.float32
            )

            lengkap = valid.copy()

            for i, name in enumerate(BANDS):

                raw = baca_band(
                    item.assets[name].href,
                    transform, width, height,
                    Resampling.bilinear
                ).astype(np.float32)

                lengkap &= np.isfinite(raw)
                lengkap &= raw > 0

                scene[i] = np.maximum(
                    (raw - 1000.0) / 10000.0,
                    0.0
                )

                print("  Band:", name, "selesai")

            scene[:, ~lengkap] = np.nan

            file_temp = TEMP / (
                f"scene_{len(dipilih) + 1:02d}.tif"
            )

            simpan_raster(
                file_temp,
                scene,
                transform,
                BANDS
            )

            info = {
                "tanggal": tanggal.isoformat(),
                "item_id": item.id,
                "awan_metadata": item.properties.get(
                    "eo:cloud_cover"
                ),
                "valid_persen": round(
                    float(lengkap.mean() * 100), 2
                ),
                "file": str(file_temp)
            }

            dipilih.append(info)
            tanggal_terpilih.append(tanggal)

            print("Citra diterima.")
            print(
                "Jumlah terpilih:",
                len(dipilih),
                "/",
                JUMLAH_TANGGAL
            )

        except Exception as exc:
            print("Citra dilewati:", str(exc)[:300])

if len(dipilih) < MIN_TANGGAL:
    raise RuntimeError(
        f"Hanya {len(dipilih)} tanggal berhasil. "
        f"Minimal {MIN_TANGGAL} tanggal diperlukan."
    )


# ============================================================
# 6. HITUNG MEDIAN TIAP BAND
# ============================================================

print("\n" + "=" * 65)
print("MENGHITUNG KOMPOSIT MEDIAN")
print("=" * 65)

komposit = np.full(
    (10, height, width),
    np.nan,
    dtype=np.float32
)

jumlah_observasi = np.zeros(
    (height, width),
    dtype=np.uint8
)

# Jumlah tanggal valid pada piksel, berdasarkan band B11.
for info in dipilih:
    with rasterio.open(info["file"]) as src:
        nilai = src.read(9)
        jumlah_observasi += np.isfinite(nilai).astype(np.uint8)

for band_idx, nama in enumerate(BANDS, 1):

    pengamatan = []

    for info in dipilih:
        with rasterio.open(info["file"]) as src:
            pengamatan.append(src.read(band_idx))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        komposit[band_idx - 1] = np.nanmedian(
            np.stack(pengamatan),
            axis=0
        ).astype(np.float32)

    print("Median", nama, "selesai.")

# Agar piksel komposit berasal dari beberapa tanggal,
# wajib ada >=2 observasi valid.
valid_komposit = jumlah_observasi >= 2

komposit[:, ~valid_komposit] = np.nan

simpan_raster(
    OUTPUT10,
    komposit,
    transform,
    BANDS
)

print("Komposit 10 band tersimpan.")


# ============================================================
# 7. MEMBUAT 19 FITUR
# ============================================================

print("\nMenghitung 19 fitur...")

b = {
    name: komposit[i]
    for i, name in enumerate(BANDS)
}

indeks = [
    rasio(
        b["B08"] - b["B04"],
        b["B08"] + b["B04"]
    ),

    rasio(
        b["B08"] - b["B11"],
        b["B08"] + b["B11"]
    ),

    rasio(
        b["B03"] - b["B11"],
        b["B03"] + b["B11"]
    ),

    rasio(
        b["B11"] - b["B08"],
        b["B11"] + b["B08"]
    ),

    rasio(
        b["B03"] - b["B08"],
        b["B03"] + b["B08"]
    ),

    rasio(
        b["B08"] - b["B03"],
        b["B11"] - b["B03"],
        minimum=0.02,
        absolut=True
    ),

    rasio(
        b["B05"] - b["B04"],
        b["B11"] - b["B03"],
        minimum=0.02
    ),

    rasio(
        b["B8A"] - b["B11"],
        b["B8A"] + b["B11"]
    ),

    rasio(
        b["B8A"] - b["B04"],
        b["B8A"] + b["B04"]
    )
]

fitur19 = np.concatenate(
    [komposit, np.stack(indeks)],
    axis=0
).astype(np.float32)

valid19 = np.isfinite(fitur19).all(axis=0)
fitur19[:, ~valid19] = np.nan

simpan_raster(
    OUTPUT19,
    fitur19,
    transform,
    FEATURES
)

print("Raster 19 fitur tersimpan.")


# ============================================================
# 8. LAPORAN
# ============================================================

total = width * height

laporan = {
    "periode": PERIODE,
    "area_bbox": BBOX,
    "resolusi_meter": RES,
    "ukuran": [width, height],
    "jumlah_citra": len(dipilih),
    "tanggal_citra": dipilih,
    "metode": "Median reflektansi multi-tanggal",
    "minimal_observasi_per_piksel": 2,
    "valid_10_band": int(valid_komposit.sum()),
    "valid_19_fitur": int(valid19.sum()),
    "persentase_valid_19": round(
        100 * valid19.sum() / total, 2
    ),
    "output_10_band": str(OUTPUT10),
    "output_19_fitur": str(OUTPUT19)
}

with LAPORAN.open("w", encoding="utf-8") as f:
    json.dump(laporan, f, indent=2, ensure_ascii=False)


print("\n" + "=" * 65)
print("TAHAP 27 BERHASIL")
print("=" * 65)

print("Jumlah tanggal:", len(dipilih))
print("Tanggal yang digunakan:")

for info in dipilih:
    print(
        " -",
        info["tanggal"],
        "| valid:",
        info["valid_persen"],
        "%"
    )

print("Ukuran:", width, "x", height)
print("Resolusi:", RES, "meter")
print("Valid komposit:", int(valid_komposit.sum()))
print("Valid 19 fitur:", int(valid19.sum()))
print(
    "Persentase valid:",
    round(100 * valid19.sum() / total, 2),
    "%"
)

print("\nOutput 10 band:", OUTPUT10)
print("Output 19 fitur:", OUTPUT19)
print("Laporan:", LAPORAN)

print("\nBelum dilakukan training Random Forest.")
print("File hasil tahap sebelumnya tidak diubah.")
