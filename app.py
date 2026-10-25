

# ============================================================

# WEBSITE UTS KLASIFIKASI TUTUPAN LAHAN JAWA TIMUR

# STREAMLIT + LEAFLET + RANDOM FOREST

# ============================================================


from pathlib import Path
from urllib.parse import urlsplit

import json
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

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
EXPERIMENT_REPORTS = REPORTS / "eksperimen_sampling_poligon"
# Prioritaskan peta 100 m berbasis model RF 19 fitur, bukan peta sampling 14 fitur.
PETA_100M_PILIHAN = [
    MAPS / "peta_interaktif_rf19_sawah_dipulihkan.html",
    MAPS / "peta_interaktif_rf19_tanpa_mangrove_palsu.html",
    MAPS / "peta_interaktif_rf19_waterbody_koreksi.html",
    MAPS / "peta_interaktif_rf19_screening.html",
    MAPS / "peta_interaktif_rf.html",
]
EXPERIMENT_MAP = MAPS / "peta_interaktif_rf_sampling_poligon.html"


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

    path = REPORTS / "eksperimen_mangrove_17fitur" / "metrics.json"


    if not path.exists():

        return None


    with open(path, "r", encoding="utf-8") as f:

        return json.load(f)


@st.cache_data

def baca_csv(nama):

    # Laporan evaluasi harus dari eksperimen; metadata umum boleh dari laporan utama.
    path = EXPERIMENT_REPORTS / nama
    if nama in ("distribusi_geojson.csv", "informasi_band.csv") and not path.exists():
        path = REPORTS / nama

    if not path.exists():

        return None


    return pd.read_csv(path)


@st.cache_data

def baca_peta():

    path = next((p for p in PETA_100M_PILIHAN if p.is_file()), None)

    if path is None:

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




@st.cache_data
def baca_laporan_20m():
    candidates = [
        REPORTS / "29_hasil_evaluasi.json",
        REPORTS / "29_laporan_training_random_forest.json",
        REPORTS / "29_evaluasi_random_forest.json",
        REPORTS / "evaluasi_rf_20m.json",
        REPORTS / "metrics_20m.json",
    ]
    for p in candidates:
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8")), str(p.relative_to(BASE))
            except (ValueError, OSError):
                pass
    return None, None

@st.cache_data
def baca_luas_20m():
    p = REPORTS / "31_laporan_mosaik_jatim.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return None
    return None

def tampil_metrics(model):
    if not model:
        st.info("File metrik belum tersedia; tidak menampilkan angka yang belum diverifikasi.")
        return
    kunci = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]
    labels = ["Accuracy", "Balanced accuracy", "Macro F1", "Weighted F1"]
    cols = st.columns(4)
    for col, label, key in zip(cols, labels, kunci):
        nilai = model.get(key)
        if isinstance(nilai, (int,float)):
            col.metric(label, f"{nilai*100:.2f}%" if nilai <= 1 else f"{nilai:.2f}%")
        else:
            col.metric(label, "—")

@st.cache_resource(show_spinner=False)
def server_peta_lokal():
    """Layanan tile lokal hidup selama proses Streamlit, bukan proses terminal terpisah."""
    folder = MAPS / "peta_interaktif_jatim"
    if not (folder / "index_kelas.html").is_file():
        return None

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(folder), **kwargs)
        def log_message(self, format, *args):
            pass

    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    except OSError as exc:
        st.error(f"Gagal menyalakan layanan tile lokal: {exc}")
        return None
    thread = threading.Thread(target=server.serve_forever, daemon=True, name="server-tile-20m")
    thread.start()
    return server


def url_peta_20m():
    # Jika tile telah di-host melalui HTTPS, atur PETA_20M_URL pada
    # Streamlit Cloud -> App settings -> Secrets.
    # Contoh: PETA_20M_URL = "https://domain.example/peta/index_kelas.html"
    try:
        url_online = str(st.secrets.get("PETA_20M_URL", "")).strip()
    except (FileNotFoundError, KeyError):
        url_online = ""
    if url_online:
        if not url_online.startswith("https://"):
            st.warning("PETA_20M_URL harus menggunakan HTTPS agar dapat dimuat pada website online.")
            return None
        return url_online

    # Mode lokal: hanya bekerja di browser pada komputer yang menjalankan
    # Streamlit. Browser di HP / Streamlit Cloud tidak dapat memakai 127.0.0.1 ini.
    srv = server_peta_lokal()
    if srv is None:
        return None
    return f"http://127.0.0.1:{srv.server_port}/index_kelas.html"


def html_peta_20m_statis():
    """Peta leaflet menggunakan PNG yang di-host oleh static serving Streamlit."""
    folder = BASE / "static" / "peta20m"
    html_path = folder / "index_kelas.html"
    if not html_path.is_file() or not (folder / "kelas_tiles").is_dir():
        return None
    try:
        alamat = urlsplit(st.context.url)
        if not alamat.scheme or not alamat.netloc:
            return None
        url_tile = f"{alamat.scheme}://{alamat.netloc}/app/static/peta20m/"
        html = html_path.read_text(encoding="utf-8")
        # URL absolut dibutuhkan karena components.html memakai iframe.
        for nama in ("builtup", "farmland", "water", "forest", "mangrove"):
            html = html.replace(f'"kelas_tiles/{nama}/', f'"{url_tile}kelas_tiles/{nama}/')
            html = html.replace(f"'kelas_tiles/{nama}/", f"'{url_tile}kelas_tiles/{nama}/")
        html = html.replace('"tiles/', f'"{url_tile}tiles/')
        html = html.replace("'tiles/", f"'{url_tile}tiles/")
        return html
    except (OSError, ValueError):
        return None


def tampil_peta_20m(tinggi):
    html = html_peta_20m_statis()
    if html:
        components.html(html, height=tinggi, scrolling=False)
        return True
    url = url_peta_20m()
    if url:
        components.iframe(url, height=tinggi, scrolling=False)
        return True
    return False


luas_20m = baca_luas_20m()
metrics_20m, sumber_metrics_20m = baca_laporan_20m()


with st.sidebar:
    foto = BASE / "assets" / "foto_profil.jpg"
    if foto.exists():
        st.image(str(foto), width=150)
    st.markdown("### Pemetaan Tutupan Lahan Jawa Timur")
    st.caption("Dua alur pengolahan Sentinel-2 • Random Forest")
    halaman = st.radio("Menu Navigasi", ["Beranda", "Business Understanding", "Data Understanding", "Data Preparation", "Random Forest", "Evaluation", "Peta Interaktif", "Kesimpulan"])
    st.divider()
    st.caption("GEE 100 m | Python 20 m | GeoJSON | Streamlit")

if halaman == "Beranda":
    st.title("Pemetaan dan Klasifikasi Tutupan Lahan Jawa Timur")
    st.write("Proyek ini menyajikan **dua hasil pengolahan citra Sentinel-2** dengan algoritma **Random Forest**: pendekatan Google Earth Engine beresolusi 100 meter dan pemrosesan lokal Python beresolusi 20 meter. Lima kelas yang dianalisis ialah Built-up, Farm Land, Water Body, Forest, dan Mangrove.")
    a,b,c,d = st.columns(4)
    a.metric("Pendekatan", "2")
    b.metric("Kelas tutupan lahan", "5")
    c.metric("Fitur peta 100 m", "19")
    d.metric("Fitur Python 20 m", "19")
    st.subheader("Dua alur penelitian")
    st.markdown("**Alur A — GEE 100 m.** Sentinel-2 SR Harmonized melalui GEE; komposit median periode Januari–September 2026; 19 fitur (10 band + 9 indeks/fitur tambahan), 200 pohon RF pada peta hasil 100 m. Evaluasi model 19 fitur tersimpan terpisah dari arsip sampling poligon 14 fitur.\n\n**Alur B — Python 20 m.** Sentinel-2 L2A melalui Microsoft Planetary Computer; komposit median pada periode Juni–September 2026 dengan 4–6 tanggal per blok; 19 fitur. Model Python diterapkan pada 195 blok lalu dimosaikkan untuk seluruh Jawa Timur.")
    st.info("Kedua peta adalah hasil prediksi model. Resolusi spasial yang lebih tinggi tidak otomatis berarti akurasi klasifikasi lebih tinggi.")

elif halaman == "Business Understanding":
    st.title("Business Understanding")
    st.header("Latar belakang")
    st.write("Jawa Timur memiliki beragam tutupan lahan, meliputi kawasan terbangun, pertanian, perairan, hutan, dan mangrove. Citra multispektral Sentinel-2 dan model Random Forest dapat digunakan untuk memetakan sebarannya.")
    st.header("Rumusan masalah")
    st.markdown("1. Bagaimana hasil klasifikasi lima kelas tutupan lahan pada pemrosesan 100 meter dan 20 meter?\n2. Bagaimana perbedaan alur data, fitur, dan performa evaluasi kedua model?\n3. Bagaimana menampilkan kedua hasil pada peta interaktif dengan lapisan yang dapat dipilih?")
    st.header("Tujuan")
    st.write("Menghasilkan dan menyajikan kedua peta dalam satu aplikasi, mendokumentasikan konfigurasi yang berbeda, serta menilai kualitasnya berdasarkan data uji dan pemeriksaan spasial.")
    st.warning("Perbandingan ini bersifat deskriptif, bukan eksperimen terkontrol pengaruh resolusi semata: data, komposit, jumlah fitur, dan konfigurasi model juga berbeda.")

elif halaman == "Data Understanding":
    st.title("Data Understanding")
    tab_a,tab_b = st.tabs(["GEE — 100 meter", "Python — 20 meter"])
    with tab_a:
        st.subheader("Sumber dan karakteristik data — alur lama")
        st.markdown("**Sumber:** Komposit Sentinel-2 SR Harmonized melalui Google Earth Engine, kemudian prediksi Random Forest dengan Python lokal.  \n**Periode:** Januari–September 2026 (komposit median).  \n**Resolusi komposit klasifikasi:** 100 meter.  \n**CRS pengolahan terkait:** EPSG:32749.  \n**Sampel eksperimen lama:** 328 poligon GeoJSON, lima kelas (72 Built-up, 80 Farm Land, 53 Water Body, 50 Forest, 73 Mangrove), menurut aplikasi terdahulu. Angka sampel ini perlu dicocokkan dengan berkas training peta 19 fitur jika ingin diklaim sebagai sampel final peta tersebut.")
        distribusi=baca_csv("distribusi_geojson.csv")
        if distribusi is not None:
            st.dataframe(distribusi,use_container_width=True,hide_index=True)
        st.caption("Angka dataset di atas milik eksperimen lama; jangan dicampur dengan data latih model 20 m.")
    with tab_b:
        st.subheader("Sumber dan karakteristik data — alur baru")
        st.markdown("**Sumber:** Sentinel-2 Level-2A, Microsoft Planetary Computer (STAC).  \n**Periode:** 1 Juni–30 September 2026.  \n**Komposit:** Median hingga enam tanggal untuk blok awal dan hingga empat tanggal untuk blok berikutnya; minimal dua pengamatan valid per piksel.  \n**Resolusi:** 20 meter.  \n**CRS:** EPSG:32749.  \n**Sampel:** GeoJSON lima kelas; dataset training akhir 34.497 piksel dari 151 poligon yang terekstraksi.  \n**Cakupan klasifikasi:** 195 blok mosaik, dipotong sesuai batas Jawa Timur.")
        st.caption("Total poligon GeoJSON master dan poligon yang berhasil diekstraksi adalah dua angka berbeda.")

elif halaman == "Data Preparation":
    st.title("Data Preparation")
    tab_a,tab_b = st.tabs(["GEE — 100 meter", "Python — 20 meter"])
    with tab_a:
        st.markdown("**Peta akhir 100 m:** komposit Sentinel-2 GEE dengan 10 band dasar, kemudian 9 indeks/fitur tambahan menjadi 19 fitur (NDVI, NDMI, MNDWI, NDBI, NDWI, MVI, REMI, MANGROVE_8A_11, VEG_8A_4). Prediksi dilakukan menggunakan model lokal `random_forest_model_mangrove_17fitur.joblib` yang terbukti menerima 19 fitur. Ada tahapan koreksi eksperimental pada peta 100 m.\n\n**Evaluasi peta 100 m:** tersedia pada eksperimen Mangrove yang mencatat **19 fitur**, sementara eksperimen sampling poligon **14 fitur** adalah arsip model lain.")
        a,b,c=st.columns(3)
        a.metric("Piksel eksperimen 14 fitur", "48.290")
        b.metric("Poligon eksperimen", "313")
        c.metric("Fitur peta akhir", "19")
        st.caption("Jumlah piksel dan poligon di atas berasal dari eksperimen 14 fitur pada app.py lama, BUKAN bukti jumlah data training untuk raster 19 fitur.")
        st.write("Pada eksperimen lama, kelas Forest dibatasi maksimal 2.000 piksel training, diambil lebih merata antarpoligon; kelas lain menggunakan sampling acak sesuai rancangan lama.")
    with tab_b:
        st.markdown("**Langkah:** mengambil citra Sentinel-2 L2A berbasis STAC; memeriksa offset reflektansi; memasker awan/bayangan dari SCL; menyelaraskan 10 band ke grid 20 m; menghitung komposit median; menurunkan 9 indeks/fitur tambahan; mengekstraksi piksel dari poligon GeoJSON; memisahkan training–testing berdasarkan ID poligon.")
        a,b,c=st.columns(3)
        a.metric("Piksel dataset", "34.497")
        b.metric("Poligon terekstraksi", "151")
        c.metric("Jumlah fitur", "19")
        st.markdown("**19 fitur:** B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12, NDVI, NDMI, MNDWI, NDBI, NDWI, MVI, REMI, MANGROVE_8A_11, VEG_8A_4.")
        st.warning("Komposit awal menggunakan hingga 6 tanggal, sedangkan blok susulan hingga 4 tanggal. Konsistensi batas antarkomposit perlu diperiksa.")

elif halaman == "Random Forest":
    st.title("Random Forest")
    st.write("Keduanya menggunakan RandomForestClassifier, tetapi model dilatih dengan dataset dan konfigurasi berbeda. Model dan evaluasinya **tidak boleh tertukar**.")
    tabel=pd.DataFrame({"Parameter":["Sumber pengolahan", "Resolusi", "Fitur", "Jumlah pohon", "Bobot kelas", "random_state", "Dataset label"],
      "GEE–Python 100 m (peta akhir)":["GEE (komposit) + Python (RF)", "100 m", "19", "200", "Periksa konfigurasi training 19 fitur", "Periksa konfigurasi training 19 fitur", "Model mangrove_17fitur (secara aktual 19 fitur)"],
      "Python 20 m":["Planetary Computer + scikit-learn", "20 m", "19", "300", "balanced_subsample", "42", "151 poligon terekstraksi; 34.497 piksel"]})
    st.dataframe(tabel,use_container_width=True,hide_index=True)
    st.markdown("**Model peta 100 m 19 fitur:** `random_forest_model_mangrove_17fitur.joblib` (nama berkas lama, jumlah fitur model yang dibaca = 19). **Model arsip evaluasi 14 fitur:** `random_forest_model_sampling_poligon.joblib`.  \n**Model baru:** `outputs/models/random_forest_20m_19fitur.joblib`, dilatih dengan `min_samples_leaf=2`.")
    st.info("Evaluasi peta 100 m menggunakan eksperimen Mangrove 19 fitur (237 poligon training, 60 testing); evaluasi 20 m menggunakan 120 poligon training dan 31 testing. Evaluasi keduanya tidak memakai set uji yang sama.")

elif halaman == "Evaluation":
    st.title("Evaluation — Perbandingan Model 100 m dan 20 m")
    st.write("Data evaluasi ditarik langsung dari dua berkas JSON yang sesuai dengan masing-masing model **19 fitur**. Evaluasi dilakukan pada kelompok sampel yang berbeda, sehingga selisih akurasi bukan bukti bahwa satu resolusi lebih akurat secara umum.")
    tab_a, tab_b, tab_c = st.tabs(["GEE–Python 100 m", "Python 20 m", "Perbandingan metrik"])
    with tab_a:
        st.caption("Sumber: outputs/reports/eksperimen_mangrove_17fitur/metrics.json. Meskipun nama folder 17fitur, JSON dan model yang dipakai mencatat 19 fitur.")
        tampil_metrics(metrics)
        if metrics:
            a, b, c = st.columns(3)
            a.metric("Training (piksel)", f"{metrics.get('jumlah_training', 0):,}")
            b.metric("Testing (piksel)", f"{metrics.get('jumlah_testing', 0):,}")
            c.metric("Poligon testing", str(metrics.get('jumlah_poligon_testing', '—')))
            st.write(f"Poligon training: **{metrics.get('jumlah_poligon_training', '—')}**. Model memakai **{metrics.get('jumlah_fitur', '—')} fitur**, dengan 200 pohon.")
            st.write("Performa khusus Mangrove pada pengujian:")
            st.write(f"Precision **{metrics.get('precision_mangrove', 0)*100:.2f}%**, recall **{metrics.get('recall_mangrove', 0)*100:.2f}%**, F1 **{metrics.get('f1_mangrove', 0)*100:.2f}%**.")
        else:
            st.warning("File evaluasi model 100 m belum ditemukan.")
    with tab_b:
        st.caption("Sumber: outputs/reports/29_hasil_evaluasi.json (model 20 m, 19 fitur, 300 pohon).")
        if metrics_20m:
            c1, c2, c3 = st.columns(3)
            c1.metric("Accuracy", f"{metrics_20m['accuracy_pixel']*100:.2f}%")
            c2.metric("Balanced accuracy", f"{metrics_20m['balanced_accuracy_pixel']*100:.2f}%")
            c3.metric("Accuracy poligon", f"{metrics_20m['accuracy_poligon']*100:.2f}%")
            a, b, c, d = st.columns(4)
            a.metric("Training (piksel)", f"{metrics_20m['jumlah_training']:,}")
            b.metric("Testing (piksel)", f"{metrics_20m['jumlah_testing']:,}")
            c.metric("Poligon training", metrics_20m['poligon_training'])
            d.metric("Poligon testing", metrics_20m['poligon_testing'])
            st.caption("Macro F1 dan weighted F1 tidak ada dalam laporan ini sehingga tidak ditampilkan sebagai angka. Accuracy poligon 100% berasal dari 31 poligon pengujian, bukan bukti akurasi 100% seluruh Jawa Timur.")
        else:
            st.warning("Laporan 29_hasil_evaluasi.json belum tersedia.")
        st.warning("Sebaran sampel testing Mangrove terbatas. Generalisasi geografis masih perlu divalidasi.")
    with tab_c:
        if metrics and metrics_20m:
            baris = pd.DataFrame({
                "Model": ["GEE–Python 100 m", "Python 20 m"],
                "Accuracy (%)": [metrics['accuracy']*100, metrics_20m['accuracy_pixel']*100],
                "Balanced accuracy (%)": [metrics['balanced_accuracy']*100, metrics_20m['balanced_accuracy_pixel']*100],
                "Jumlah fitur": [metrics.get('jumlah_fitur', 19), 19],
                "Pohon RF": [200, 300],
                "Testing piksel": [metrics['jumlah_testing'], metrics_20m['jumlah_testing']],
                "Testing poligon": [metrics['jumlah_poligon_testing'], metrics_20m['poligon_testing']],
            })
            st.dataframe(baris, hide_index=True, use_container_width=True)
            st.bar_chart(baris.set_index("Model")[["Accuracy (%)", "Balanced accuracy (%)"]], stack=False)
            st.write(f"Selisih accuracy pengujian: **{(metrics['accuracy'] - metrics_20m['accuracy_pixel'])*100:.2f} poin persentase**, lebih tinggi pada set uji model 100 m. Namun data latih, pembagian testing, komposit, dan parameter model tidak identik.")
        else:
            st.warning("Perbandingan memerlukan kedua file metrik JSON.")
    with st.expander("Arsip eksperimen lain (bukan metrik kedua peta final)"):
        st.caption("Eksperimen sampling poligon 14 fitur disimpan terpisah di outputs/reports/eksperimen_sampling_poligon/metrics.json; tidak dipakai pada perbandingan utama.")

elif halaman == "Peta Interaktif":
    st.title("Peta Interaktif Tutupan Lahan Jawa Timur")
    st.write("Dua peta prediksi ditampilkan terpisah berdasarkan alur pengolahannya. Checkbox tiap kelas tersedia pada peta 20 meter.")
    html_100=baca_peta()
    path_100=next((p for p in PETA_100M_PILIHAN if p.is_file()), None)
    path_20=MAPS/"peta_interaktif_jatim"/"index_kelas.html"
    tab_a,tab_b,tab_c=st.tabs(["Hasil eksperimen 100 m", "Hasil Python 20 m", "Perbandingan berdampingan"])
    with tab_a:
        st.subheader("Peta GEE–Python 100 m — pastikan versi raster sesuai")
        if html_100 is not None:
            components.html(html_100,height=730,scrolling=False)
            st.caption(f"Berkas peta 100 m: {path_100.name}. Pastikan varian koreksi yang dipilih sesuai hasil akhir yang diinginkan.")
        else:
            st.warning("Peta RF 100 m 19 fitur belum ditemukan. Periksa berkas peta_interaktif_rf19_*.html atau peta_interaktif_rf.html dalam outputs/maps. File sampling 14 fitur tidak otomatis dijadikan peta 19 fitur.")
    with tab_b:
        st.subheader("Random Forest Python — 20 m, 19 fitur")
        if tampil_peta_20m(730):
            st.caption("Peta 20 m: lima layer dapat diaktifkan/dinonaktifkan secara terpisah.")
        else:
            st.warning("Peta 20 m belum siap. Jalankan SIAPKAN_PETA_20M.ps1 untuk menyiapkan folder static/peta20m.")
    with tab_c:
        a,b=st.columns(2)
        with a:
            st.markdown("**100 m — GEE**")
            if html_100: components.html(html_100,height=560,scrolling=False)
            else: st.warning("Peta eksperimen belum ada")
        with b:
            st.markdown("**20 m — Python**")
            if not tampil_peta_20m(560): st.warning("Peta 20 m belum ada")
    st.warning("Klasifikasi merupakan hasil prediksi. Kesalahan label di wilayah perairan, hutan, dan mangrove masih perlu diperiksa secara geografis.")

elif halaman == "Kesimpulan":
    st.title("Kesimpulan Dua Pendekatan Klasifikasi")
    st.subheader("GEE–Python 100 meter — 19 fitur")
    st.write("Komposit Sentinel-2 melalui Google Earth Engine digunakan pada grid 100 meter, kemudian diklasifikasikan memakai Random Forest Python dengan 200 pohon dan 19 fitur. Peta 100 m melewati sejumlah tahap koreksi eksperimental yang perlu dilaporkan secara terpisah dari prediksi murni.")
    if metrics:
        st.success(f"Hasil pengujian model 100 m: accuracy **{metrics['accuracy']*100:.2f}%**, balanced accuracy **{metrics['balanced_accuracy']*100:.2f}%**, dan macro F1 **{metrics['macro_f1']*100:.2f}%** pada {metrics['jumlah_testing']:,} piksel testing.")
    else:
        st.info("Laporan evaluasi 100 m belum ditemukan.")
    st.subheader("Python 20 meter — 19 fitur")
    st.write("Sentinel-2 L2A melalui Microsoft Planetary Computer dikompositkan pada grid 20 meter. Model Random Forest 300 pohon digunakan untuk klasifikasi 195 blok dan pembuatan mosaik Jawa Timur.")
    if metrics_20m:
        st.success(f"Hasil pengujian model 20 m: accuracy **{metrics_20m['accuracy_pixel']*100:.2f}%** dan balanced accuracy **{metrics_20m['balanced_accuracy_pixel']*100:.2f}%** pada {metrics_20m['jumlah_testing']:,} piksel testing.")
    if luas_20m:
        st.subheader("Luas kelas hasil prediksi 20 m")
        st.caption("Laporan mosaik adalah estimasi dari prediksi kelas, bukan statistik tutupan lahan resmi.")
        with st.expander("Tampilkan laporan luas lengkap"):
            st.json(luas_20m)
    st.subheader("Interpretasi")
    st.write("Resolusi 20 meter menghasilkan detail spasial lebih tinggi daripada 100 meter. Pada data testing masing-masing, angka accuracy model 100 m lebih tinggi, tetapi **perbandingan ini bukan uji terkendali efek resolusi**: citra komposit, tanggal, data latih, split testing, dan parameter Random Forest tidak sama.")
    st.subheader("Keterbatasan")
    st.markdown("- Kedua model masih memerlukan pengujian dengan titik referensi independen yang mewakili seluruh Jawa Timur.\n- Prediksi Water Body dan Mangrove di lokasi tertentu masih perlu pemeriksaan geografis.\n- Komposit 20 m menggunakan campuran empat dan enam tanggal pada blok berbeda.\n- Hasil 100 m memuat koreksi eksperimental sehingga perlu dibedakan dari prediksi RF tanpa koreksi.\n- Angka luas klasifikasi tidak boleh dianggap data tutupan lahan resmi tanpa validasi.")
