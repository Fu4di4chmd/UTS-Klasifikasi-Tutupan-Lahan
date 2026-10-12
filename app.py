

# ============================================================

# WEBSITE UTS KLASIFIKASI TUTUPAN LAHAN JAWA TIMUR

# STREAMLIT + LEAFLET + RANDOM FOREST

# ============================================================


from pathlib import Path

import json

import pandas as pd

import streamlit as st

import streamlit.components.v1 as components


# ============================================================

# 1. KONFIGURASI WEBSITE

# ============================================================


st.set_page_config(

    page_title="Klasifikasi Tutupan Lahan Jawa Timur",


    layout="wide",

    initial_sidebar_state="expanded"

)


BASE = Path(__file__).resolve().parent

REPORTS = BASE / "outputs" / "reports"

CHARTS = BASE / "outputs" / "charts"

MAPS = BASE / "outputs" / "maps"


# ============================================================

# 2. CUSTOM CSS

# ============================================================


st.markdown("""

<style>

    .block-container {

        padding-top: 1.6rem;

        padding-bottom: 3rem;

        max-width: 1250px;

    }


    h1, h2, h3 {

        letter-spacing: -0.5px;

    }


    div[data-testid="stMetric"] {

        border: 1px solid rgba(128,128,128,0.25);

        border-radius: 12px;

        padding: 16px;

    }


    .info-box {

        border-left: 4px solid #228B22;

        background: rgba(128,128,128,0.07);

        padding: 16px 20px;

        border-radius: 8px;

        margin-bottom: 16px;

    }

</style>

""", unsafe_allow_html=True)


# ============================================================

# 3. BACA HASIL PENELITIAN

# ============================================================


@st.cache_data

def baca_metrics():

    path = REPORTS / "metrics.json"


    if not path.exists():

        return None


    with open(path, "r", encoding="utf-8") as f:

        return json.load(f)


@st.cache_data

def baca_csv(nama):

    path = REPORTS / nama


    if not path.exists():

        return None


    return pd.read_csv(path)


@st.cache_data

def baca_peta():

    path = MAPS / "peta_interaktif.html"


    if not path.exists():

        return None


    return path.read_text(encoding="utf-8")


def tampil_grafik(nama_file, judul):

    path = CHARTS / nama_file


    st.subheader(judul)


    if path.exists():

        st.image(str(path), use_container_width=True)

    else:

        st.warning(f"Grafik belum ditemukan: {nama_file}")


metrics = baca_metrics()


# ============================================================

# 4. NAVIGASI SIDEBAR

# ============================================================


with st.sidebar:

    # PROFIL PEMBUAT
    foto_path = BASE / "assets" / "foto_profil.jpg"

    if foto_path.exists():
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            st.image(str(foto_path), width="stretch")

    st.markdown(
        """
        <div style="text-align:center; margin-top:12px;">
            <h3 style="margin-bottom:5px;">
                Achmad Saiful Fuadi
            </h3>
            <p style="font-size:13px; color:gray;">
                Pemetaan dan Klasifikasi Tutupan Lahan
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    halaman = st.radio(
        "Menu Navigasi",
        [
            "Beranda",
            "Business Understanding",
            "Data Understanding",
            "Data Preparation",
            "Random Forest",
            "Evaluation",
            "Peta Interaktif",
            "Kesimpulan"
        ]
    )

    st.divider()

    st.caption("Teknologi")
    st.write("Python • Sentinel-2 • Random Forest")
    st.write("GeoJSON • Streamlit • Leaflet")
# ============================================================

# 5. BERANDA

# ============================================================


if halaman == "Beranda":


    st.title("Pemetaan dan Klasifikasi Tutupan Lahan")


    st.subheader("Wilayah Kajian Jawa Timur")


    st.markdown("""

    Proyek ini mengintegrasikan citra satelit **Sentinel-2**,

    data **GeoJSON hasil digitasi**, dan algoritma

    **Random Forest** untuk mengidentifikasi karakteristik

    lima kelas tutupan lahan.


    Data spasial divisualisasikan menggunakan **Leaflet**

    agar pengguna dapat mengamati sebaran poligon dan

    memilih kelas tutupan lahan yang ingin ditampilkan.

    """)


    st.divider()


    c1, c2, c3, c4 = st.columns(4)


    c1.metric("Poligon GeoJSON", "302")

    c2.metric("Kelas Tutupan Lahan", "5")

    c3.metric("Fitur Klasifikasi", "14")


    if metrics:

        c4.metric(

            "Akurasi Random Forest",

            f"{metrics['accuracy'] * 100:.2f}%"

        )

    else:

        c4.metric("Akurasi Random Forest", "Belum tersedia")


    st.divider()


    st.header("Tentang Proyek")


    st.markdown("""

    Tutupan lahan menggambarkan kondisi permukaan bumi

    seperti kawasan terbangun, lahan pertanian, perairan,

    hutan, dan hutan mangrove.


    Melalui pengolahan citra satelit, karakteristik

    spektral setiap kelas dapat dipelajari oleh algoritma

    machine learning.


    Dalam proyek ini, data digitasi digunakan sebagai

    sampel berlabel untuk training dan testing Random

    Forest. Hasilnya kemudian dievaluasi dan disajikan

    bersama peta interaktif.

    """)


    st.header("Alur Pengerjaan")


    st.markdown("""

    **1. Pengambilan Data** — Citra Sentinel-2 dan GeoJSON


    **2. Pengolahan Data** — Validasi, ekstraksi piksel,

    serta perhitungan indeks spektral


    **3. Klasifikasi** — Random Forest menggunakan 14 fitur


    **4. Evaluasi** — Accuracy, balanced accuracy,

    F1-score, dan confusion matrix


    **5. Penyajian Informasi** — Peta Leaflet pada Streamlit

    """)


    st.info(

        "Peta GeoJSON menunjukkan hasil digitasi manual. "

        "Nilai akurasi berasal dari evaluasi prediksi "

        "Random Forest pada data testing."

    )


# ============================================================

# 6. BUSINESS UNDERSTANDING

# ============================================================


elif halaman == "Business Understanding":


    st.title("Business Understanding")


    st.header("Latar Belakang")


    st.write("""

    Wilayah Jawa Timur memiliki kondisi tutupan lahan

    yang beragam, mulai dari kawasan perkotaan,

    pertanian, perairan, hutan, hingga hutan mangrove.


    Keberagaman tersebut memerlukan informasi spasial

    yang dapat membantu pengamatan persebaran dan

    karakteristik tutupan lahan.


    Citra Sentinel-2 menyediakan informasi spektral

    untuk membedakan jenis tutupan lahan, sedangkan

    Random Forest digunakan untuk mempelajari hubungan

    antara nilai spektral dan label kelas.

    """)


    st.header("Tujuan Utama")


    st.markdown("""

    <div class="info-box">

    Mengidentifikasi karakteristik lima kelas tutupan

    lahan di wilayah kajian Jawa Timur menggunakan

    citra Sentinel-2 dan Random Forest, serta

    memvisualisasikan data hasil digitasi melalui

    peta interaktif berbasis Leaflet.

    </div>

    """, unsafe_allow_html=True)


    st.header("Tujuan Khusus")


    st.markdown("""

    1. Membuat data GeoJSON untuk lima kategori tutupan lahan.

    2. Mengolah sepuluh band spektral Sentinel-2.

    3. Menghasilkan empat indeks spektral tambahan.

    4. Melatih Random Forest menggunakan sampel berlabel.

    5. Menguji kemampuan model pada data testing.

    6. Menampilkan GeoJSON melalui peta interaktif.

    7. Menyajikan hasil pengolahan dalam website Streamlit.

    """)


    st.header("Manfaat")


    st.write("""

    Proyek ini menunjukkan pemanfaatan penginderaan jauh

    dan machine learning untuk analisis tutupan lahan.


    Informasi yang diperoleh dapat digunakan sebagai

    media pembelajaran pemetaan digital, interpretasi

    citra satelit, dan analisis data geospasial.

    """)


# ============================================================

# 7. DATA UNDERSTANDING

# ============================================================


elif halaman == "Data Understanding":


    st.title("Data Understanding")


    st.header("Sumber Data")


    st.markdown("""

    **Citra:** Sentinel-2 Level-2A


    **Sumber:** Copernicus Browser


    **Tanggal citra:** 1 Oktober 2026


    **Wilayah:** Area kajian di Jawa Timur


    **CRS raster:** EPSG:32749


    **Resolusi TIFF hasil ekspor:** sekitar 93,44 meter/piksel


    **Data referensi:** 302 poligon GeoJSON hasil digitasi

    """)


    st.header("Distribusi Kelas GeoJSON")


    distribusi = baca_csv("distribusi_geojson.csv")


    if distribusi is not None:

        st.dataframe(

            distribusi,

            use_container_width=True,

            hide_index=True

        )

    else:

        st.warning("Data distribusi GeoJSON tidak ditemukan.")


    st.header("Lima Kelas Tutupan Lahan")


    kelas = pd.DataFrame({

        "Kelas": [

            "Built-up",

            "Farmland",

            "Water Body",

            "Forest",

            "Mangrove"

        ],

        "Definisi": [

            "Kawasan terbangun dan permukiman",

            "Lahan pertanian termasuk sawah",

            "Laut, danau, sungai, dan perairan",

            "Hutan daratan non-mangrove",

            "Vegetasi mangrove di wilayah pesisir"

        ],

        "Jumlah Poligon": [72, 80, 50, 50, 50]

    })


    st.dataframe(

        kelas,

        hide_index=True,

        use_container_width=True

    )


    st.header("Band Sentinel-2")


    bands = baca_csv("informasi_band.csv")


    if bands is not None:

        st.dataframe(

            bands,

            use_container_width=True,

            hide_index=True

        )


    st.subheader("Fokus Band B8A, B11, dan B04")


    st.markdown("""

    - **B8A (Narrow NIR):** Berkaitan dengan respons vegetasi.

    - **B11 (SWIR):** Sensitif terhadap kondisi kelembapan.

    - **B04 (Red):** Membantu membedakan karakteristik vegetasi.


    Ketiga band dapat dikombinasikan untuk membantu

    interpretasi hutan dan mangrove. Pembedaan keduanya

    juga memerlukan perhatian pada lokasi pesisir dan

    karakteristik lingkungan.

    """)


# ============================================================

# 8. DATA PREPARATION

# ============================================================


elif halaman == "Data Preparation":


    st.title("Data Preparation")


    st.write("""

    Data Preparation dilakukan untuk menyiapkan

    nilai spektral citra Sentinel-2 dan label GeoJSON

    agar dapat digunakan dalam klasifikasi Random Forest.

    """)


    st.header("Tahapan Pengolahan")


    st.markdown("""

    **1. Pemeriksaan raster:** Menyesuaikan CRS, dimensi,

    dan transformasi sepuluh band.


    **2. Pemeriksaan GeoJSON:** Memastikan label kelas dan

    geometri poligon valid.


    **3. Ekstraksi piksel:** Mengambil nilai sepuluh band

    pada lokasi poligon hasil digitasi.


    **4. Validasi nilai:** Mengeluarkan piksel yang tidak

    memenuhi kriteria validitas numerik.


    **5. Perhitungan indeks spektral:** NDVI, NDMI, MNDWI,

    dan NDBI.


    **6. Pembagian data:** Training dan testing dipisahkan

    berdasarkan ID poligon.

    """)


    st.header("Indeks Spektral")


    indeks = pd.DataFrame({

        "Indeks": ["NDVI", "NDMI", "MNDWI", "NDBI"],

        "Kegunaan": [

            "Mengamati kehijauan vegetasi",

            "Mengamati karakteristik kelembapan",

            "Membantu identifikasi perairan",

            "Membantu identifikasi kawasan terbangun"

        ]

    })


    st.dataframe(

        indeks,

        hide_index=True,

        use_container_width=True

    )


    st.header("Hasil Preprocessing")


    c1, c2, c3 = st.columns(3)


    c1.metric("Sampel Piksel Valid", "55.008")

    c2.metric("Poligon Aktif", "299")

    c3.metric("Jumlah Fitur", "14")


    st.write("""

    Dari 302 poligon awal, terdapat 299 poligon

    yang menghasilkan sampel piksel valid.

    Tiga poligon lainnya tidak menghasilkan sampel

    piksel yang memenuhi kriteria pengolahan.

    """)


    st.header("Pembagian Training dan Testing")


    if metrics:

        a, b = st.columns(2)


        a.metric(

            "Poligon Training",

            metrics["jumlah_poligon_training"]

        )


        b.metric(

            "Poligon Testing",

            metrics["jumlah_poligon_testing"]

        )


        st.write(

            f"Training setelah balancing: "

            f"{metrics['jumlah_training']:,} sampel piksel."

        )


        st.write(

            f"Testing: "

            f"{metrics['jumlah_testing']:,} sampel piksel."

        )


    st.success(

        "Training dan testing dipisahkan berdasarkan "

        "poligon sehingga tidak ada ID poligon "

        "yang digunakan pada kedua kelompok."

    )


# ============================================================

# 9. RANDOM FOREST

# ============================================================


elif halaman == "Random Forest":


    st.title("Klasifikasi Random Forest")


    st.header("Pengertian")


    st.write("""

    Random Forest merupakan algoritma machine learning

    yang menggabungkan banyak pohon keputusan untuk

    menghasilkan prediksi kelas.


    Setiap sampel piksel direpresentasikan oleh sepuluh

    nilai band Sentinel-2 dan empat indeks spektral.

    """)


    st.header("Konfigurasi Model")


    konfigurasi = pd.DataFrame({

        "Parameter": [

            "Algoritma",

            "Jumlah pohon",

            "Jumlah fitur",

            "max_features",

            "class_weight",

            "random_state"

        ],

        "Nilai": [

            "RandomForestClassifier",

            "200",

            "14",

            "sqrt",

            "balanced",

            "42"

        ]

    })


    st.dataframe(

        konfigurasi,

        hide_index=True,

        use_container_width=True

    )


    st.header("Penyeimbangan Sampel")


    st.write("""

    Pada data awal, kelas Forest memiliki jumlah

    piksel jauh lebih besar daripada kelas lainnya.


    Untuk mengurangi dominasi tersebut, jumlah sampel

    training dibatasi maksimal 2.000 piksel per kelas.


    Strategi ini hanya diterapkan pada training.

    Data testing tetap dipertahankan.

    """)


    st.header("Alur Model")


    st.markdown("""

    **Nilai 14 fitur**

    → **Training Random Forest**

    → **Prediksi data testing**

    → **Perbandingan dengan label GeoJSON**

    → **Evaluasi kinerja model**

    """)


    st.info(

        "Hasil model disimpan dalam file "

        "random_forest_model.joblib."

    )


# ============================================================

# 10. EVALUATION

# ============================================================


elif halaman == "Evaluation":


    st.title("Evaluasi Random Forest")


    if metrics is None:

        st.error("metrics.json belum ditemukan.")

        st.stop()


    c1, c2, c3, c4 = st.columns(4)


    c1.metric(

        "Accuracy",

        f"{metrics['accuracy'] * 100:.2f}%"

    )


    c2.metric(

        "Balanced Accuracy",

        f"{metrics['balanced_accuracy'] * 100:.2f}%"

    )


    c3.metric(

        "Macro F1",

        f"{metrics['macro_f1'] * 100:.2f}%"

    )


    c4.metric(

        "Weighted F1",

        f"{metrics['weighted_f1'] * 100:.2f}%"

    )


    st.divider()


    tampil_grafik(

        "evaluasi_model.png",

        "Perbandingan Metrik Evaluasi"

    )


    st.divider()


    tampil_grafik(

        "confusion_matrix.png",

        "Confusion Matrix"

    )


    st.divider()


    tampil_grafik(

        "performa_per_kelas.png",

        "Performa Setiap Kelas"

    )


    st.divider()


    tampil_grafik(

        "feature_importance.png",

        "Feature Importance"

    )


    st.divider()


    tampil_grafik(

        "distribusi_sampel.png",

        "Distribusi Sampel Training dan Testing"

    )


    st.divider()


    st.header("Classification Report")


    report = baca_csv("classification_report.csv")


    if report is not None:

        st.dataframe(

            report,

            use_container_width=True,

            hide_index=True

        )


    st.header("Interpretasi")


    st.markdown(f"""

    Model Random Forest memperoleh akurasi sebesar

    **{metrics['accuracy'] * 100:.2f}%**

    berdasarkan pengujian pada

    **{metrics['jumlah_testing']:,} sampel piksel**.


    Balanced accuracy sebesar

    **{metrics['balanced_accuracy'] * 100:.2f}%**

    menunjukkan rata-rata kemampuan model mengenali

    setiap kelas.


    Macro F1-score sebesar

    **{metrics['macro_f1'] * 100:.2f}%**

    menunjukkan bahwa performa klasifikasi

    antarkelas belum sepenuhnya merata.

    """)


# ============================================================

# 11. PETA INTERAKTIF

# ============================================================


elif halaman == "Peta Interaktif":


    st.title("Peta Interaktif Tutupan Lahan")


    st.write("""

    Peta menampilkan **302 poligon GeoJSON hasil

    digitasi** di atas citra Sentinel-2 True Color.


    Gunakan checkbox pada sudut kanan atas peta

    untuk menampilkan atau menyembunyikan masing-masing

    kelas tutupan lahan.

    """)


    html = baca_peta()


    if html is None:

        st.error(

            "Peta HTML tidak ditemukan. "

            "Jalankan scripts/05_create_map.py terlebih dahulu."

        )

    else:

        components.html(

            html,

            height=780,

            scrolling=False

        )


    st.caption(

        "Sumber: Sentinel-2 Copernicus Browser "

        "dan GeoJSON hasil digitasi."

    )


    st.info(

        "Warna poligon pada peta adalah label "

        "hasil digitasi manual. Peta ini tidak "

        "menampilkan prediksi Random Forest "

        "untuk seluruh wilayah."

    )


    path_peta = MAPS / "peta_interaktif.html"


    if path_peta.exists():

        st.download_button(

            label="Download Peta HTML",

            data=path_peta.read_bytes(),

            file_name="peta_tutupan_lahan_jatim.html",

            mime="text/html"

        )


# ============================================================

# 12. KESIMPULAN

# ============================================================


elif halaman == "Kesimpulan":


    st.title("Kesimpulan Proyek")


    st.markdown("""

    Proyek ini menerapkan penginderaan jauh

    dan algoritma Random Forest untuk

    mengklasifikasikan sampel tutupan lahan

    pada wilayah kajian Jawa Timur.


    Data yang digunakan terdiri atas 302 poligon

    GeoJSON dengan lima kelas tutupan lahan.

    Pengolahan citra Sentinel-2 menghasilkan

    sepuluh fitur band spektral dan empat

    indeks spektral tambahan.


    Model dilatih dan diuji menggunakan pembagian

    berdasarkan ID poligon untuk mencegah

    penggunaan poligon yang sama pada kedua kelompok.

    """)


    if metrics:

        st.success(

            f"Random Forest menghasilkan accuracy "

            f"{metrics['accuracy'] * 100:.2f}% "

            f"dan balanced accuracy "

            f"{metrics['balanced_accuracy'] * 100:.2f}%."

        )


    st.header("Informasi yang Diperoleh")


    st.markdown("""

    - Lokasi sampel lima kelas tutupan lahan dapat

      diamati melalui peta satelit.

    - Random Forest dapat mempelajari karakteristik

      spektral dari 14 fitur.

    - Kinerja model dapat dievaluasi menggunakan

      beberapa metrik.

    - Leaflet memungkinkan pengguna memfilter

      kelas tutupan lahan secara interaktif.

    """)


    st.header("Keterbatasan")


    st.markdown("""

    - Citra TIFF memiliki resolusi ekspor sekitar

      93,44 meter per piksel.

    - Sebagian area citra masih menunjukkan awan.

    - Beberapa poligon kecil tidak menghasilkan

      piksel valid.

    - Distribusi sampel antarjenis tutupan lahan

      tidak seimbang.

    - Peta GeoJSON adalah visualisasi sampel

      digitasi, bukan peta prediksi menyeluruh.

    """)


    st.header("Pengembangan Selanjutnya")


    st.write("""

    Pengembangan dapat dilakukan menggunakan

    citra dengan resolusi spasial lebih tinggi,

    penyaringan awan, penambahan sampel pada

    kelas yang sulit dikenali, serta pengujian

    hasil klasifikasi pada wilayah yang berbeda.

    """)
