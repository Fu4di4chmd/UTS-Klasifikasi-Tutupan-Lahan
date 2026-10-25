
"""
TAHAP 28D - EKSTRAKSI SAMPEL RF 19 FITUR, 20 METER

Input:
- Raster komposit Tahap 28C
- GeoJSON sampel hasil Tahap 28B
- Konfigurasi blok Tahap 28B

Output:
- CSV piksel training percobaan
- CSV statistik per poligon
- JSON laporan ekstraksi

Catatan:
- Membaca seluruh 19 fitur.
- Mempertahankan ID poligon asli.
- Memotong geometri sesuai batas blok.
- Maksimal 300 piksel per poligon.
- Tidak melakukan training.
- Tidak menimpa raster.

Jalankan:
py -3.9 scripts/28d_ekstraksi_sampel_20m.py
"""

from pathlib import Path
from collections import Counter
import csv
import json

import numpy as np
import rasterio

from rasterio.features import geometry_mask
from rasterio.windows import from_bounds, Window
from rasterio.warp import transform_geom
from shapely.geometry import shape, mapping, box


# ============================================================
# 1. KONFIGURASI
# ============================================================

BASE = Path(__file__).resolve().parent.parent

KONFIG = BASE / "outputs/reports/tahap28f_B_28_456.json"

GEOJSON = BASE / "data/geojson/tahap28f_B_28_456.geojson"

OUT = BASE / "outputs/datasets"
OUT.mkdir(parents=True, exist_ok=True)

MAX_PIXEL_PER_POLIGON = 300
SEED = 42

KELAS = {
    "build up": 1,
    "farm land": 2,
    "water body": 3,
    "forest": 4,
    "mangrove": 5
}

FEATURES = [
    "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B11", "B12",
    "NDVI", "NDMI", "MNDWI", "NDBI",
    "NDWI", "MVI", "REMI",
    "MANGROVE_8A_11", "VEG_8A_4"
]

rng = np.random.default_rng(SEED)

with KONFIG.open("r", encoding="utf-8") as f:
    konfigurasi = json.load(f)

BLOK_ID = konfigurasi["blok_terpilih"]

RASTER = (
    BASE
    / "data/prediksi/sentinel2_20m_blok"
    / BLOK_ID
    / "komposit_sentinel2_19fitur_20m.tif"
)

OUTPUT_CSV = OUT / f"tahap28d_sampel_{BLOK_ID}.csv"
OUTPUT_STAT = OUT / f"tahap28d_statistik_{BLOK_ID}.csv"
OUTPUT_JSON = OUT / f"tahap28d_laporan_{BLOK_ID}.json"

for path in [KONFIG, GEOJSON, RASTER]:
    if not path.exists():
        raise FileNotFoundError(path)


# ============================================================
# 2. MEMBACA POLIGON
# ============================================================

with GEOJSON.open("r", encoding="utf-8") as f:
    geojson = json.load(f)

features = geojson.get("features", [])

if not features:
    raise RuntimeError("GeoJSON sampel kosong.")

print("=" * 65)
print("TAHAP 28D - EKSTRAKSI SAMPEL 19 FITUR, 20 METER")
print("=" * 65)
print("Blok:", BLOK_ID)
print("Poligon pada GeoJSON:", len(features))
print("Raster:", RASTER.name)


# ============================================================
# 3. EKSTRAKSI PIXEL PER POLIGON
# ============================================================

statistik = []
jumlah_kelas = Counter()

total_valid = 0
total_terpilih = 0
total_overlap = 0

xmin, ymin, xmax, ymax = konfigurasi["bbox_utm"]
batas_blok = box(xmin, ymin, xmax, ymax)

with rasterio.open(RASTER) as src:

    if src.count != 19:
        raise ValueError("Raster harus memiliki 19 band.")

    if list(src.descriptions) != FEATURES:
        raise ValueError("Urutan 19 fitur tidak sesuai.")

    if src.crs is None:
        raise ValueError("Raster tidak mempunyai CRS.")

    if src.crs.to_epsg() != 32749:
        raise ValueError(
            "Grid blok belum memakai EPSG:32749."
        )

    if (
        abs(src.res[0] - 20) > 0.01
        or abs(src.res[1] - 20) > 0.01
    ):
        raise ValueError("Raster bukan grid 20 meter.")

    # Mendata piksel yang sudah dimiliki poligon lain.
    # Piksel tumpang tindih tidak dipakai dua kali.
    occupied = np.zeros(
        (src.height, src.width),
        dtype=bool
    )

    headers = [
        "blok_id",
        "id_poligon",
        "kelas",
        "kode_kelas",
        "pixel_row",
        "pixel_col",
        "x_utm",
        "y_utm"
    ] + FEATURES

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file_csv:

        writer = csv.DictWriter(
            file_csv,
            fieldnames=headers
        )
        writer.writeheader()

        for urutan, ft in enumerate(features, start=1):

            props = ft.get("properties") or {}

            id_poligon = int(
                props["id_sampel_asli"]
            )

            kelas = str(
                props.get("class", "")
            ).strip().lower()

            if kelas not in KELAS:
                raise ValueError(
                    f"Kelas tidak dikenal: {kelas}"
                )

            geom_asli = ft.get("geometry")

            if not geom_asli:
                raise ValueError(
                    f"Geometri kosong: {id_poligon}"
                )

            geom_utm = shape(
                transform_geom(
                    "EPSG:4326",
                    src.crs,
                    geom_asli
                )
            )

            # Potong dengan batas blok dan cakupan raster.
            geom_clip = geom_utm.intersection(
                batas_blok
            ).intersection(
                box(*src.bounds)
            )

            if geom_clip.is_empty:
                statistik.append({
                    "id_poligon": id_poligon,
                    "kelas": kelas,
                    "piksel_dalam_poligon": 0,
                    "piksel_valid": 0,
                    "piksel_overlap": 0,
                    "piksel_diambil": 0
                })
                continue

            try:
                l, b, r, t = geom_clip.bounds

                win = from_bounds(
                    l, b, r, t,
                    transform=src.transform
                )

                # Ambil seluruh sel yang bersinggungan
                # dengan area geometri.
                col0 = max(
                    0, int(np.floor(win.col_off))
                )
                row0 = max(
                    0, int(np.floor(win.row_off))
                )

                col1 = min(
                    src.width,
                    int(np.ceil(win.col_off + win.width))
                )
                row1 = min(
                    src.height,
                    int(np.ceil(win.row_off + win.height))
                )

                if col1 <= col0 or row1 <= row0:
                    continue

                window = Window(
                    col0, row0,
                    col1 - col0,
                    row1 - row0
                )

                data = src.read(window=window)

                inside = geometry_mask(
                    [mapping(geom_clip)],
                    out_shape=(
                        int(window.height),
                        int(window.width)
                    ),
                    transform=src.window_transform(window),
                    invert=True,
                    all_touched=False
                )

                finite = np.isfinite(data).all(axis=0)

                count_inside = int(inside.sum())
                count_valid = int(
                    (inside & finite).sum()
                )

                occupied_local = occupied[
                    row0:row1, col0:col1
                ]

                overlap_mask = (
                    inside & finite & occupied_local
                )

                count_overlap = int(
                    overlap_mask.sum()
                )

                eligible = (
                    inside
                    & finite
                    & (~occupied_local)
                )

                rr, cc = np.where(eligible)

                # Agar model tidak terlalu dipengaruhi
                # oleh poligon yang sangat luas.
                if len(rr) > MAX_PIXEL_PER_POLIGON:

                    chosen = rng.choice(
                        len(rr),
                        MAX_PIXEL_PER_POLIGON,
                        replace=False
                    )

                    rr = rr[chosen]
                    cc = cc[chosen]

                # Tetapkan semua piksel valid dalam
                # poligon sebagai terpakai, bukan hanya
                # yang terpilih secara acak.
                occupied_local[inside & finite] = True

                untuk_csv = []

                for local_r, local_c in zip(rr, cc):

                    row_global = row0 + int(local_r)
                    col_global = col0 + int(local_c)

                    x, y = src.xy(
                        row_global,
                        col_global
                    )

                    nilai = data[
                        :, local_r, local_c
                    ]

                    record = {
                        "blok_id": BLOK_ID,
                        "id_poligon": id_poligon,
                        "kelas": kelas,
                        "kode_kelas": KELAS[kelas],
                        "pixel_row": row_global,
                        "pixel_col": col_global,
                        "x_utm": round(x, 2),
                        "y_utm": round(y, 2)
                    }

                    for nama, value in zip(
                        FEATURES, nilai
                    ):
                        record[nama] = float(value)

                    untuk_csv.append(record)

                writer.writerows(untuk_csv)

                statistik.append({
                    "id_poligon": id_poligon,
                    "kelas": kelas,
                    "piksel_dalam_poligon": count_inside,
                    "piksel_valid": count_valid,
                    "piksel_overlap": count_overlap,
                    "piksel_diambil": len(untuk_csv)
                })

                total_valid += count_valid
                total_terpilih += len(untuk_csv)
                total_overlap += count_overlap
                jumlah_kelas[kelas] += len(untuk_csv)

                print(
                    f"[{urutan}/{len(features)}]",
                    f"Poligon {id_poligon}",
                    f"| {kelas}",
                    f"| valid: {count_valid}",
                    f"| dipilih: {len(untuk_csv)}"
                )

            except Exception as exc:
                raise RuntimeError(
                    f"Ekstraksi gagal pada poligon "
                    f"{id_poligon}: {exc}"
                ) from exc


# ============================================================
# 4. SIMPAN STATISTIK
# ============================================================

with OUTPUT_STAT.open(
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "id_poligon",
            "kelas",
            "piksel_dalam_poligon",
            "piksel_valid",
            "piksel_overlap",
            "piksel_diambil"
        ]
    )

    writer.writeheader()
    writer.writerows(statistik)

poligon_aktif = sum(
    row["piksel_diambil"] > 0
    for row in statistik
)

laporan = {
    "blok": BLOK_ID,
    "raster": str(RASTER),
    "resolusi_meter": 20,
    "jumlah_fitur": 19,
    "jumlah_poligon_input": len(features),
    "poligon_dengan_sampel": poligon_aktif,
    "total_piksel_valid_dalam_poligon": total_valid,
    "total_piksel_overlap": total_overlap,
    "total_piksel_terpilih": total_terpilih,
    "maks_piksel_per_poligon": MAX_PIXEL_PER_POLIGON,
    "jumlah_piksel_per_kelas": {
        k: jumlah_kelas[k]
        for k in KELAS
    },
    "catatan": (
        "Dataset ini hanya berasal dari satu blok "
        "dan belum siap melatih model 5 kelas. "
        "ID poligon asli dipertahankan. "
        "Sampling dilakukan tanpa penggantian."
    )
}

with OUTPUT_JSON.open(
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        laporan,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# 5. RINGKASAN
# ============================================================

print("\n" + "=" * 65)
print("TAHAP 28D BERHASIL")
print("=" * 65)

print("Blok:", BLOK_ID)
print("Poligon input:", len(features))
print("Poligon dengan sampel:", poligon_aktif)
print("Piksel valid dalam poligon:", total_valid)
print("Piksel overlap:", total_overlap)
print("Total piksel diambil:", total_terpilih)

print("\nDISTRIBUSI SAMPEL PIXEL")

for kelas in KELAS:
    print(
        f"{kelas:12s}:",
        jumlah_kelas[kelas]
    )

print("\nOUTPUT")
print("Dataset:", OUTPUT_CSV)
print("Statistik:", OUTPUT_STAT)
print("Laporan:", OUTPUT_JSON)

print("\nBelum dilakukan training Random Forest.")
