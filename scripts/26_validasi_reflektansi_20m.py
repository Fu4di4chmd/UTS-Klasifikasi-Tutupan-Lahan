
"""
TAHAP 26 - VALIDASI REFLEKTANSI SENTINEL-2

Memeriksa:
1. Kesesuaian grid dan jumlah piksel valid.
2. Nilai DN dan reflektansi.
3. Perhitungan 19 fitur.
4. Metadata BOA_ADD_OFFSET dan QUANTIFICATION_VALUE.

Tidak mengubah file raster.

Jalankan:
py -3.9 scripts/26_validasi_reflektansi_20m.py
"""

from pathlib import Path
import json
import xml.etree.ElementTree as ET

import numpy as np
import rasterio
from pystac_client import Client
import planetary_computer


BASE = Path(__file__).resolve().parent.parent
FOLDER = BASE / "data/prediksi/sentinel2_20m_uji"

INPUT10 = FOLDER / "uji_sentinel2_10band_20m.tif"
INPUT19 = FOLDER / "uji_sentinel2_19fitur_20m.tif"

OUTPUT = FOLDER / "laporan_validasi_tahap26.json"

BANDS = [
    "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B11", "B12"
]

FEATURES = BANDS + [
    "NDVI", "NDMI", "MNDWI", "NDBI",
    "NDWI", "MVI", "REMI",
    "MANGROVE_8A_11", "VEG_8A_4"
]


def local_name(tag):
    return tag.split("}")[-1]


def find_values(root, name):
    result = []
    for node in root.iter():
        if local_name(node.tag) == name:
            if node.text and node.text.strip():
                result.append(node.text.strip())
    return result


def ratio(n, d, absolute=False):
    if absolute:
        n = np.abs(n)
        d = np.abs(d)

    out = np.full(n.shape, np.nan, dtype=np.float32)
    valid = (
        np.isfinite(n)
        & np.isfinite(d)
        & (np.abs(d) > 1e-6)
    )
    np.divide(n, d, out=out, where=valid)
    return out


print("=" * 65)
print("TAHAP 26 - VALIDASI REFLEKTANSI DAN 19 FITUR")
print("=" * 65)

for file in [INPUT10, INPUT19]:
    if not file.exists():
        raise FileNotFoundError(file)

with rasterio.open(INPUT10) as src10, \
     rasterio.open(INPUT19) as src19:

    if src10.count != 10 or src19.count != 19:
        raise ValueError("Jumlah band tidak sesuai.")

    if list(src10.descriptions) != BANDS:
        raise ValueError("Urutan 10 band tidak sesuai.")

    if list(src19.descriptions) != FEATURES:
        raise ValueError("Urutan 19 fitur tidak sesuai.")

    if (
        src10.shape != src19.shape
        or src10.transform != src19.transform
        or src10.crs != src19.crs
    ):
        raise ValueError("Grid raster tidak cocok.")

    dn = src10.read().astype(np.float32)
    hasil = src19.read().astype(np.float32)

    tags = src10.tags()
    item_id = tags.get("stac_item")

    valid10 = (
        np.isfinite(dn).all(axis=0)
        & (dn > 0).all(axis=0)
    )

    valid19 = np.isfinite(hasil).all(axis=0)

    total = valid10.size

    print("\n--- DATA RASTER ---")
    print("Grid:", src10.res)
    print("CRS:", src10.crs)
    print("Ukuran:", src10.width, "x", src10.height)
    print("Total piksel:", total)
    print("Valid 10 band:", int(valid10.sum()))
    print("Valid 19 fitur:", int(valid19.sum()))
    print(
        "Persentase valid 19 fitur:",
        round(100 * valid19.sum() / total, 2),
        "%"
    )
    print("ID citra:", item_id)

    metadata = {
        "stac_item": item_id,
        "metadata_terverifikasi": False,
        "boa_offset": None,
        "quantification_value": None
    }

    # Periksa metadata produk yang digunakan.
    if item_id:
        try:
            print("\n--- MEMERIKSA METADATA PRODUK ---")

            catalog = Client.open(
                "https://planetarycomputer.microsoft.com/api/stac/v1",
                modifier=planetary_computer.sign_inplace
            )

            items = list(catalog.search(
                collections=["sentinel-2-l2a"],
                ids=[item_id],
                max_items=1
            ).items())

            if items:
                item = items[0]

                if "product-metadata" in item.assets:
                    import urllib.request

                    url = item.assets["product-metadata"].href

                    with urllib.request.urlopen(
                        url, timeout=60
                    ) as response:
                        xml_bytes = response.read()

                    root = ET.fromstring(xml_bytes)

                    offsets = find_values(
                        root, "BOA_ADD_OFFSET"
                    )

                    quant = find_values(
                        root, "BOA_QUANTIFICATION_VALUE"
                    )

                    print("BOA_ADD_OFFSET:", offsets)
                    print("BOA_QUANTIFICATION_VALUE:", quant)

                    if offsets and quant:
                        values = [float(x) for x in offsets]
                        q = float(quant[0])

                        metadata["boa_offset"] = values
                        metadata["quantification_value"] = q

                        if (
                            all(abs(v + 1000) < 1e-6 for v in values)
                            and abs(q - 10000) < 1e-6
                        ):
                            metadata["metadata_terverifikasi"] = True

        except Exception as exc:
            print("Metadata online belum dapat diperiksa:", exc)

    if not metadata["metadata_terverifikasi"]:
        print(
            "\nPERINGATAN: Offset belum terverifikasi. "
            "Jangan gunakan raster untuk training final."
        )
    else:
        print("\nOffset dan kuantifikasi sesuai Tahap 25.")

    # Rekalkulasi dengan nilai yang dipakai Tahap 25.
    reflektansi = (dn - 1000.0) / 10000.0
    reflektansi[:, ~valid10] = np.nan

    b = {name: reflektansi[i] for i, name in enumerate(BANDS)}

    indeks = [
        ratio(b["B08"] - b["B04"], b["B08"] + b["B04"]),
        ratio(b["B08"] - b["B11"], b["B08"] + b["B11"]),
        ratio(b["B03"] - b["B11"], b["B03"] + b["B11"]),
        ratio(b["B11"] - b["B08"], b["B11"] + b["B08"]),
        ratio(b["B03"] - b["B08"], b["B03"] + b["B08"]),
        ratio(
            b["B08"] - b["B03"],
            b["B11"] - b["B03"],
            absolute=True
        ),
        ratio(b["B05"] - b["B04"], b["B11"] - b["B03"]),
        ratio(b["B8A"] - b["B11"], b["B8A"] + b["B11"]),
        ratio(b["B8A"] - b["B04"], b["B8A"] + b["B04"])
    ]

    perhitungan = np.concatenate(
        [reflektansi, np.stack(indeks)]
    )

    sama = np.isclose(
        perhitungan,
        hasil,
        rtol=1e-4,
        atol=1e-5,
        equal_nan=True
    )

    cocok = bool(sama.all())

    print("\n--- VALIDASI HASIL PERHITUNGAN ---")
    print("Seluruh 19 fitur cocok:", cocok)

    statistik = {}

    for i, name in enumerate(FEATURES):
        nilai = hasil[i]
        finite = nilai[np.isfinite(nilai)]

        if len(finite) == 0:
            statistik[name] = {"valid": 0}
            continue

        statistik[name] = {
            "valid": int(len(finite)),
            "min": float(np.min(finite)),
            "median": float(np.median(finite)),
            "max": float(np.max(finite))
        }

        print(
            f"{name:16s}",
            "valid:", len(finite),
            "min:", round(float(np.min(finite)), 4),
            "median:", round(float(np.median(finite)), 4),
            "max:", round(float(np.max(finite)), 4)
        )

    # Indeks normalisasi idealnya berada dalam [-1, 1]
    # untuk reflektansi nonnegatif. Reflektansi negatif
    # dapat membuat indeks keluar dari rentang tersebut.
    indeks_normalisasi = [
        "NDVI", "NDMI", "MNDWI", "NDBI", "NDWI",
        "MANGROVE_8A_11", "VEG_8A_4"
    ]

    peringatan = []

    for name in indeks_normalisasi:
        nilai = hasil[FEATURES.index(name)]
        finite = nilai[np.isfinite(nilai)]

        if len(finite):
            luar = int(np.sum(np.abs(finite) > 1.0))
            if luar:
                peringatan.append(
                    f"{name}: {luar} piksel di luar [-1, 1]"
                )

    laporan = {
        "total_piksel": int(total),
        "valid_10_band": int(valid10.sum()),
        "valid_19_fitur": int(valid19.sum()),
        "persentase_valid": round(
            100 * valid19.sum() / total, 4
        ),
        "metadata": metadata,
        "perhitungan_19_fitur_cocok": cocok,
        "peringatan_indeks": peringatan,
        "statistik": statistik
    }

    with OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(
            laporan, f, indent=2, ensure_ascii=False
        )

    print("\n" + "=" * 65)
    print("HASIL VALIDASI TAHAP 26")
    print("=" * 65)
    print("Metadata diverifikasi:",
          metadata["metadata_terverifikasi"])
    print("19 fitur cocok:", cocok)
    print("Peringatan indeks:", len(peringatan))
    print("Laporan:", OUTPUT)

    if not cocok:
        print("STATUS: PERLU PERBAIKAN PERHITUNGAN")
    elif not metadata["metadata_terverifikasi"]:
        print("STATUS: PERLU VERIFIKASI METADATA")
    else:
        print("STATUS: VALIDASI DASAR BERHASIL")
        print(
            "Catatan: validasi ini belum membuktikan "
            "akurasi klasifikasi atau kualitas sampel."
        )
