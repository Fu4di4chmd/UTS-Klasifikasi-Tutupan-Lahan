
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)

BASE = Path(__file__).resolve().parents[1]
INPUT = BASE / "outputs/datasets/dataset_final_20m_19fitur.csv"
OUTPUT = BASE / "outputs" / "models"
LAPORAN = BASE / "outputs" / "reports"
OUTPUT.mkdir(parents=True, exist_ok=True)
LAPORAN.mkdir(parents=True, exist_ok=True)

FITUR = [
    "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B11", "B12",
    "NDVI", "NDMI", "MNDWI", "NDBI",
    "NDWI", "MVI", "REMI",
    "MANGROVE_8A_11", "VEG_8A_4"
]

KELAS = [
    "build up", "farm land", "water body",
    "forest", "mangrove"
]

print("=" * 65)
print("TAHAP 29 - TRAINING DAN EVALUASI RANDOM FOREST")
print("=" * 65)

df = pd.read_csv(INPUT)
df["kelas"] = df["kelas"].str.strip().str.lower()

# Setiap ID poligon harus mempunyai satu kelas saja.
cek = df.groupby("id_poligon")["kelas"].nunique()
if (cek > 1).any():
    raise ValueError("Ada ID poligon dengan lebih dari satu kelas.")

if not np.isfinite(df[FITUR].to_numpy(dtype=float)).all():
    raise ValueError("Terdapat fitur tidak valid.")

# Pemisahan di tingkat poligon, bukan piksel.
poligon = df[["id_poligon", "kelas"]].drop_duplicates()

train_ids, test_ids = train_test_split(
    poligon["id_poligon"],
    test_size=0.20,
    random_state=42,
    stratify=poligon["kelas"]
)

train = df[df["id_poligon"].isin(train_ids)].copy()
test = df[df["id_poligon"].isin(test_ids)].copy()

assert set(train["id_poligon"]).isdisjoint(
    set(test["id_poligon"])
)

X_train = train[FITUR]
y_train = train["kelas"]
X_test = test[FITUR]
y_test = test["kelas"]

print("\nDISTRIBUSI TRAINING")
print(train.groupby("kelas").agg(
    piksel=("kelas", "size"),
    poligon=("id_poligon", "nunique")
).to_string())

print("\nDISTRIBUSI TESTING")
print(test.groupby("kelas").agg(
    piksel=("kelas", "size"),
    poligon=("id_poligon", "nunique")
).to_string())

print("\nMelatih Random Forest...")

model = RandomForestClassifier(
    n_estimators=300,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
prediksi = model.predict(X_test)

print("\nHASIL EVALUASI PER PIKSEL")
print("Accuracy:", round(
    accuracy_score(y_test, prediksi), 4
))
print("Balanced accuracy:", round(
    balanced_accuracy_score(y_test, prediksi), 4
))
print(classification_report(
    y_test, prediksi, labels=KELAS,
    zero_division=0
))

cm = confusion_matrix(y_test, prediksi, labels=KELAS)
cm_df = pd.DataFrame(cm, index=KELAS, columns=KELAS)

# Evaluasi tambahan: satu keputusan per poligon testing.
test["prediksi"] = prediksi
per_poligon = (
    test.groupby("id_poligon")
    .agg(
        aktual=("kelas", "first"),
        prediksi=("prediksi", lambda s: s.value_counts().idxmax()),
        jumlah_piksel=("kelas", "size")
    )
    .reset_index()
)

print("\nHASIL EVALUASI PER POLIGON")
print("Accuracy:", round(accuracy_score(
    per_poligon["aktual"], per_poligon["prediksi"]
), 4))
print(classification_report(
    per_poligon["aktual"], per_poligon["prediksi"],
    labels=KELAS, zero_division=0
))

penting = pd.DataFrame({
    "fitur": FITUR,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

# Simpan seluruh hasil.
joblib.dump({
    "model": model,
    "fitur": FITUR,
    "kelas": KELAS
}, OUTPUT / "random_forest_20m_19fitur.joblib")

cm_df.to_csv(LAPORAN / "29_confusion_matrix.csv")
penting.to_csv(LAPORAN / "29_feature_importance.csv", index=False)
per_poligon.to_csv(LAPORAN / "29_evaluasi_poligon.csv", index=False)

train[["id_poligon", "kelas"]].drop_duplicates().to_csv(
    LAPORAN / "29_poligon_training.csv", index=False
)
test[["id_poligon", "kelas"]].drop_duplicates().to_csv(
    LAPORAN / "29_poligon_testing.csv", index=False
)

with open(LAPORAN / "29_hasil_evaluasi.json", "w") as f:
    json.dump({
        "accuracy_pixel": float(accuracy_score(y_test, prediksi)),
        "balanced_accuracy_pixel": float(
            balanced_accuracy_score(y_test, prediksi)
        ),
        "accuracy_poligon": float(accuracy_score(
            per_poligon["aktual"], per_poligon["prediksi"]
        )),
        "jumlah_training": int(len(train)),
        "jumlah_testing": int(len(test)),
        "poligon_training": int(train["id_poligon"].nunique()),
        "poligon_testing": int(test["id_poligon"].nunique())
    }, f, indent=2)

print("\nMODEL DAN LAPORAN BERHASIL DISIMPAN")
print("Tahap 29 - training dan evaluasi awal selesai.")
