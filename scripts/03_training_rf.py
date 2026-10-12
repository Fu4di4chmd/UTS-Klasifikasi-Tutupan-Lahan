
# ============================================================
# TAHAP 3 - TRAINING RANDOM FOREST
# KLASIFIKASI TUTUPAN LAHAN JAWA TIMUR
# ============================================================

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# 1. LOKASI FOLDER
# ============================================================

BASE = Path(__file__).resolve().parent.parent

REPORT_DIR = BASE / "outputs" / "reports"
MODEL_DIR = BASE / "outputs" / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = REPORT_DIR / "training.csv"
TEST_PATH = REPORT_DIR / "testing.csv"

# ============================================================
# 2. DAFTAR FITUR DAN KELAS
# ============================================================

BANDS = [
    "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B11", "B12"
]

INDEKS = [
    "NDVI", "NDMI", "MNDWI", "NDBI"
]

FEATURES = BANDS + INDEKS

CLASSES = [
    "build up",
    "farm land",
    "water body",
    "forest",
    "mangrove"
]

# ============================================================
# 3. BACA DATA TRAINING DAN TESTING
# ============================================================

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print("=" * 60)
print("TRAINING RANDOM FOREST")
print("=" * 60)

print("\nData training awal:", train_df.shape)
print("Data testing:", test_df.shape)

print("\nDistribusi training sebelum balancing:")
print(train_df["label"].value_counts())

print("\nDistribusi testing:")
print(test_df["label"].value_counts())

# ============================================================
# 4. PERIKSA DATA
# ============================================================

required_columns = FEATURES + ["polygon_id", "label"]

for col in required_columns:
    if col not in train_df.columns:
        raise ValueError(f"Kolom {col} tidak ada di training.")

    if col not in test_df.columns:
        raise ValueError(f"Kolom {col} tidak ada di testing.")

# Pastikan tidak ada kebocoran poligon
train_ids = set(train_df["polygon_id"].unique())
test_ids = set(test_df["polygon_id"].unique())

assert train_ids.isdisjoint(test_ids), (
    "Terjadi kebocoran poligon antara training dan testing!"
)

assert np.isfinite(
    train_df[FEATURES].to_numpy()
).all(), "Training masih memiliki NaN atau infinity."

assert np.isfinite(
    test_df[FEATURES].to_numpy()
).all(), "Testing masih memiliki NaN atau infinity."

# ============================================================
# 5. BALANCING DATA TRAINING
# ============================================================

# Maksimal 2.000 piksel per kelas.
# Jika jumlahnya kurang, seluruh piksel tetap digunakan.
# Sampling hanya dilakukan pada TRAINING.

MAX_SAMPLES = 2000

balanced_parts = []

for kelas in CLASSES:

    bagian = train_df[
        train_df["label"] == kelas
    ]

    if bagian.empty:
        raise ValueError(
            f"Kelas {kelas} tidak mempunyai sampel training."
        )

    jumlah = min(len(bagian), MAX_SAMPLES)

    sampel = bagian.sample(
        n=jumlah,
        random_state=42
    )

    balanced_parts.append(sampel)

balanced_df = pd.concat(
    balanced_parts,
    ignore_index=True
)

balanced_df = balanced_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

print("\n" + "=" * 60)
print("DISTRIBUSI TRAINING SETELAH BALANCING")
print("=" * 60)

print(balanced_df["label"].value_counts())

print("\nTotal training setelah balancing:", len(balanced_df))
print("Total testing tetap:", len(test_df))

balanced_df.to_csv(
    REPORT_DIR / "training_balanced.csv",
    index=False
)

# ============================================================
# 6. SIAPKAN X DAN Y
# ============================================================

X_train = balanced_df[FEATURES].to_numpy(
    dtype=np.float32
)

y_train = balanced_df["label"].to_numpy()

X_test = test_df[FEATURES].to_numpy(
    dtype=np.float32
)

y_test = test_df["label"].to_numpy()

print("\nBentuk data:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

# ============================================================
# 7. TRAINING RANDOM FOREST
# ============================================================

print("\nMelatih Random Forest...")

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_features="sqrt",
    min_samples_leaf=1,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

print("✅ Training Random Forest berhasil.")

# ============================================================
# 8. PREDIKSI DATA TESTING
# ============================================================

y_pred = rf_model.predict(X_test)

# ============================================================
# 9. EVALUASI MODEL
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

balanced_accuracy = balanced_accuracy_score(
    y_test, y_pred
)

macro_f1 = f1_score(
    y_test, y_pred,
    labels=CLASSES,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    y_test, y_pred,
    labels=CLASSES,
    average="weighted",
    zero_division=0
)

print("\n" + "=" * 60)
print("HASIL EVALUASI RANDOM FOREST")
print("=" * 60)

print(f"Accuracy         : {accuracy * 100:.2f}%")
print(f"Balanced Accuracy: {balanced_accuracy * 100:.2f}%")
print(f"Macro F1-Score   : {macro_f1 * 100:.2f}%")
print(f"Weighted F1-Score: {weighted_f1 * 100:.2f}%")

print("\nCLASSIFICATION REPORT")
print(
    classification_report(
        y_test,
        y_pred,
        labels=CLASSES,
        target_names=CLASSES,
        digits=4,
        zero_division=0
    )
)

print("\nCONFUSION MATRIX")
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=CLASSES
)

print(pd.DataFrame(
    cm,
    index=CLASSES,
    columns=CLASSES
))

# ============================================================
# 10. FEATURE IMPORTANCE
# ============================================================

importance_df = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": rf_model.feature_importances_
}).sort_values(
    "Importance",
    ascending=False
)

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

print(importance_df.to_string(index=False))

# ============================================================
# 11. SIMPAN HASIL EVALUASI
# ============================================================

report_df = pd.DataFrame(
    classification_report(
        y_test,
        y_pred,
        labels=CLASSES,
        target_names=CLASSES,
        output_dict=True,
        zero_division=0
    )
).transpose()

report_df.to_csv(
    REPORT_DIR / "classification_report.csv"
)

importance_df.to_csv(
    REPORT_DIR / "feature_importance.csv",
    index=False
)

pd.DataFrame(
    cm,
    index=CLASSES,
    columns=CLASSES
).to_csv(
    REPORT_DIR / "confusion_matrix.csv"
)

# Simpan prediksi testing beserta ID poligon
hasil_test = test_df[
    ["polygon_id", "label"]
].copy()

hasil_test["prediksi_rf"] = y_pred

hasil_test.to_csv(
    REPORT_DIR / "hasil_prediksi_testing.csv",
    index=False
)

metrics = {
    "accuracy": float(accuracy),
    "balanced_accuracy": float(balanced_accuracy),
    "macro_f1": float(macro_f1),
    "weighted_f1": float(weighted_f1),
    "jumlah_training": int(len(balanced_df)),
    "jumlah_testing": int(len(test_df)),
    "jumlah_fitur": int(len(FEATURES)),
    "jumlah_poligon_training": int(len(train_ids)),
    "jumlah_poligon_testing": int(len(test_ids))
}

with open(
    REPORT_DIR / "metrics.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(metrics, f, indent=4)

# ============================================================
# 12. SIMPAN MODEL RANDOM FOREST
# ============================================================

joblib.dump(
    {
        "model": rf_model,
        "features": FEATURES,
        "classes": CLASSES
    },
    MODEL_DIR / "random_forest_model.joblib"
)

print("\n" + "=" * 60)
print("✅ TRAINING DAN EVALUASI SELESAI")
print("=" * 60)

print("Model:", MODEL_DIR / "random_forest_model.joblib")
print("Laporan:", REPORT_DIR)
print("✅ Data testing tidak diubah.")
print("✅ Tidak ada kebocoran poligon.")
