
from pathlib import Path
from collections import Counter
import csv
import json

import joblib
import numpy as np
import rasterio
from rasterio.windows import Window

BASE = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    BASE / "outputs/models/random_forest_20m_19fitur.joblib"
)

INPUT_DIR = BASE / "data/prediksi/sentinel2_20m_blok"
OUTPUT_DIR = BASE / "outputs/maps/klasifikasi_20m"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BLOK = [
    "B_32_459",
    "B_34_459",
    "B_34_460",
    "B_28_459",
    "B_28_456",
]

FITUR = [
    "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B11", "B12",
    "NDVI", "NDMI", "MNDWI", "NDBI",
    "NDWI", "MVI", "REMI",
    "MANGROVE_8A_11", "VEG_8A_4"
]

# Kode 0 khusus untuk NoData.
KODE_KELAS = {
    "build up": 1,
    "farm land": 2,
    "water body": 3,
    "forest": 4,
    "mangrove": 5,
}

NAMA_KELAS = {
    0: "NoData",
    1: "Built-up",
    2: "Farm Land",
    3: "Water Body",
    4: "Forest",
    5: "Mangrove",
}

WARNA = {
    0: (0, 0, 0, 0),
    1: (220, 45, 45, 255),
    2: (242, 196, 65, 255),
    3: (35, 116, 225, 255),
    4: (35, 135, 65, 255),
    5: (135, 55, 165, 255),
}

WINDOW_SIZE = 512

print("=" * 65)
print("TAHAP 30 - KLASIFIKASI RANDOM FOREST 20 M")
print("=" * 65)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model tidak ditemukan: {MODEL_PATH}"
    )

paket = joblib.load(MODEL_PATH)
model = paket["model"]

if paket["fitur"] != FITUR:
    raise ValueError(
        "Urutan fitur model berbeda dari fitur raster."
    )

kelas_model = set(map(str, model.classes_))
if kelas_model != set(KODE_KELAS):
    raise ValueError(
        f"Kelas model tidak sesuai: {kelas_model}"
    )

laporan_semua = []

for i, blok in enumerate(BLOK, start=1):

    print("\n" + "=" * 65)
    print(f"[{i}/{len(BLOK)}] BLOK {blok}")
    print("=" * 65)

    input_path = (
        INPUT_DIR / blok /
        "komposit_sentinel2_19fitur_20m.tif"
    )

    output_path = (
        OUTPUT_DIR / f"klasifikasi_rf_{blok}_20m.tif"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Raster belum tersedia: {input_path}"
        )

    jumlah = Counter()

    with rasterio.open(input_path) as src:

        if src.count != len(FITUR):
            raise ValueError(
                f"{blok}: jumlah band {src.count}, "
                f"seharusnya {len(FITUR)}."
            )

        if abs(src.res[0] - 20) > 0.01 or \
           abs(src.res[1] - 20) > 0.01:
            raise ValueError(
                f"{blok}: resolusi bukan 20 meter."
            )

        if src.crs is None:
            raise ValueError(
                f"{blok}: CRS tidak ditemukan."
            )

        print("Ukuran:", src.width, "x", src.height)
        print("CRS:", src.crs)
        print("Resolusi:", src.res)

        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            dtype="uint8",
            count=1,
            nodata=0,
            compress="deflate",
            predictor=1,
            tiled=True,
            blockxsize=256,
            blockysize=256,
            BIGTIFF="IF_SAFER"
        )

        with rasterio.open(
            output_path, "w", **profile
        ) as dst:

            dst.write_colormap(1, WARNA)

            for row in range(
                0, src.height, WINDOW_SIZE
            ):
                for col in range(
                    0, src.width, WINDOW_SIZE
                ):

                    tinggi = min(
                        WINDOW_SIZE, src.height - row
                    )
                    lebar = min(
                        WINDOW_SIZE, src.width - col
                    )

                    window = Window(
                        col, row, lebar, tinggi
                    )

                    # Masked array mempertahankan
                    # NoData dari raster sumber.
                    data = src.read(
                        window=window,
                        masked=True
                    )

                    values = np.asarray(
                        data.data, dtype=np.float32
                    )

                    masks = np.ma.getmaskarray(data)

                    # Semua 19 band harus valid.
                    valid = ~np.any(masks, axis=0)
                    valid &= np.all(
                        np.isfinite(values), axis=0
                    )

                    hasil = np.zeros(
                        (tinggi, lebar),
                        dtype=np.uint8
                    )

                    if np.any(valid):

                        X = values[:, valid].T
                        prediksi = model.predict(X)

                        hasil[valid] = np.array(
                            [
                                KODE_KELAS[str(k)]
                                for k in prediksi
                            ],
                            dtype=np.uint8
                        )

                    dst.write(
                        hasil, 1, window=window
                    )

                    kode, hitung = np.unique(
                        hasil, return_counts=True
                    )

                    jumlah.update({
                        int(k): int(n)
                        for k, n in zip(kode, hitung)
                    })

            dst.update_tags(
                MODEL="Random Forest",
                FITUR="19",
                RESOLUSI="20 meter",
                SUMBER="Sentinel-2 L2A multi-tanggal",
                KELAS_1="Built-up",
                KELAS_2="Farm Land",
                KELAS_3="Water Body",
                KELAS_4="Forest",
                KELAS_5="Mangrove"
            )

    total_valid = sum(
        jumlah[k] for k in range(1, 6)
    )

    print("\nHASIL KLASIFIKASI")
    print(f"Piksel valid: {total_valid:,}")

    statistik = {}

    for kode in range(1, 6):

        n = jumlah[kode]
        persen = (
            100 * n / total_valid
            if total_valid else 0
        )

        statistik[NAMA_KELAS[kode]] = {
            "piksel": n,
            "persentase": round(persen, 3)
        }

        print(
            f"{NAMA_KELAS[kode]:12s} : "
            f"{n:>10,} ({persen:.2f}%)"
        )

    print(f"NoData       : {jumlah[0]:>10,}")

    laporan_semua.append({
        "blok": blok,
        "raster_output": str(output_path),
        "piksel_valid": total_valid,
        "piksel_nodata": jumlah[0],
        "kelas": statistik
    })

    print("Raster tersimpan:", output_path)

laporan_path = (
    OUTPUT_DIR / "laporan_klasifikasi_5blok.json"
)

with open(
    laporan_path, "w", encoding="utf-8"
) as f:
    json.dump(
        laporan_semua,
        f,
        indent=2,
        ensure_ascii=False
    )

print("\n" + "=" * 65)
print("KLASIFIKASI LIMA BLOK SELESAI")
print("=" * 65)
print("Folder hasil:", OUTPUT_DIR)
print("Laporan:", laporan_path)
print(
    "Catatan: raster ini baru mencakup "
    "lima blok, belum seluruh Jawa Timur."
)
