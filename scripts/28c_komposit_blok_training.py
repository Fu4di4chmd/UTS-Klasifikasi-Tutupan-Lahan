
"""
TAHAP 28C - KOMPOSIT BLOK TRAINING 20 METER

Menggunakan pipeline Tahap 27 yang sudah berhasil.
BBOX otomatis berasal dari hasil Tahap 28B.

Jalankan:
py -3.9 scripts/28c_komposit_blok_training.py
"""

from pathlib import Path
import json
import subprocess
import sys

BASE = Path(__file__).resolve().parent.parent

KONFIG = (
    BASE / "outputs/reports/tahap28b_blok_terpilih.json"
)

SCRIPT_ASLI = (
    BASE / "scripts/27_komposit_multitanggal_20m.py"
)

SCRIPT_BARU = (
    BASE / "scripts/28c_pipeline_blok_terpilih.py"
)

if not KONFIG.exists():
    raise FileNotFoundError(KONFIG)

if not SCRIPT_ASLI.exists():
    raise FileNotFoundError(SCRIPT_ASLI)

with KONFIG.open("r", encoding="utf-8") as f:
    config = json.load(f)

blok_id = config["blok_terpilih"]
bbox = config["bbox_lonlat"]

print("=" * 65, flush=True)
print("TAHAP 28C - KOMPOSIT BLOK 20 METER", flush=True)
print("=" * 65, flush=True)
print("Blok:", blok_id, flush=True)
print("BBOX:", bbox, flush=True)

kode = SCRIPT_ASLI.read_text(encoding="utf-8")

# Ganti direktori output menjadi folder khusus blok.
lokasi_lama = (
    'OUT = BASE / "data/prediksi/sentinel2_20m_uji"'
)

lokasi_baru = (
    f'OUT = BASE / "data/prediksi/sentinel2_20m_blok/{blok_id}"'
)

bbox_lama = 'BBOX = [112.65, -7.30, 112.85, -7.10]'
bbox_baru = f"BBOX = {bbox!r}"

if kode.count(lokasi_lama) != 1:
    raise RuntimeError(
        "Definisi OUT pada Tahap 27 tidak sesuai."
    )

if kode.count(bbox_lama) != 1:
    raise RuntimeError(
        "Definisi BBOX pada Tahap 27 tidak sesuai."
    )

kode = kode.replace(lokasi_lama, lokasi_baru, 1)
kode = kode.replace(bbox_lama, bbox_baru, 1)

# Periksa sintaks sebelum dieksekusi.
compile(kode, str(SCRIPT_BARU), "exec")

SCRIPT_BARU.write_text(kode, encoding="utf-8")

print("Script khusus blok berhasil dibuat.", flush=True)
print("Memulai komposit multi-tanggal...", flush=True)

subprocess.run(
    [sys.executable, "-u", str(SCRIPT_BARU)],
    cwd=str(BASE),
    check=True
)

print("\n" + "=" * 65)
print("TAHAP 28C BERHASIL")
print("=" * 65)

hasil = (
    BASE
    / "data/prediksi/sentinel2_20m_blok"
    / blok_id
    / "komposit_sentinel2_19fitur_20m.tif"
)

print("Raster 19 fitur:", hasil)

if not hasil.exists():
    raise FileNotFoundError(
        "Komposit belum ditemukan setelah proses."
    )
