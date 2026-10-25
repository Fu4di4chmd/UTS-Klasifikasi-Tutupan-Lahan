
"""
TAHAP 10 - PETA INTERAKTIF RF 19 FITUR TANPA MANGROVE TAMBAHAN
Wilayah: Jawa Timur
Model: RF 19 fitur + koreksi eksperimental; Mangrove dikembalikan ke prediksi asli
Raster sumber asli: 100 meter
Grid tampilan: 50 meter (nearest neighbor)

Jalankan:
py -3.9 scripts/10_peta_interaktif_rf.py
"""

from pathlib import Path
from io import BytesIO
import base64
import json

import folium
import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject


# ============================================================
# 1. LOKASI FILE
# ============================================================

BASE = Path(__file__).resolve().parent.parent

RASTER_RF = (
    BASE / "outputs" / "maps"
    / "klasifikasi_rf19_koreksi_tanpa_mangrove_palsu_50m.tif"
)

RASTER_S2 = (
    BASE / "data" / "prediksi"
    / "sentinel2_jatim_14fitur_100m.tif"
)

GEOJSON_DIR = BASE / "data" / "geojson"

OUTPUT = (
    BASE / "outputs" / "maps"
    / "peta_interaktif_rf19_tanpa_mangrove_palsu.html"
)


# ============================================================
# 2. KELAS TUTUPAN LAHAN
# ============================================================

KELAS = {
    1: ("Built-up", "#d45b56", "build up"),
    2: ("Farm land", "#edaa63", "farm land"),
    3: ("Water body", "#298fe8", "water body"),
    4: ("Forest", "#208653", "forest"),
    5: ("Mangrove", "#04bfb4", "mangrove"),
}


# ============================================================
# 3. FUNGSI GAMBAR
# ============================================================

def to_data_url(array):
    buffer = BytesIO()

    Image.fromarray(array).save(
        buffer,
        format="PNG",
        optimize=True
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("ascii")

    return "data:image/png;base64," + encoded


# ============================================================
# 4. PROYEKSI UNTUK LEAFLET
# ============================================================

def reprojection_grid(src, max_width=2400):

    transform, width, height = calculate_default_transform(
        src.crs,
        "EPSG:4326",
        src.width,
        src.height,
        *src.bounds
    )

    ratio = min(1.0, max_width / width)

    new_width = max(1, int(round(width * ratio)))
    new_height = max(1, int(round(height * ratio)))

    transform = transform * rasterio.Affine.scale(
        width / new_width,
        height / new_height
    )

    west = transform.c
    north = transform.f

    east = west + new_width * transform.a
    south = north + new_height * transform.e

    bounds = [
        [south, west],
        [north, east]
    ]

    return transform, new_width, new_height, bounds


def warper(src, band, transform, width, height, nodata):

    out = np.full(
        (height, width),
        nodata,
        dtype=np.float32
    )

    reproject(
        source=rasterio.band(src, band),
        destination=out,
        src_transform=src.transform,
        src_crs=src.crs,
        src_nodata=src.nodata,
        dst_transform=transform,
        dst_crs="EPSG:4326",
        dst_nodata=nodata,
        resampling=Resampling.nearest
    )

    return out


# ============================================================
# 5. BASEMAP SENTINEL-2
# ============================================================

def make_satellite_true_color(peta):

    if not RASTER_S2.exists():
        print("Sentinel-2 lokal tidak ditemukan.")
        return

    print("Membuat citra Sentinel-2 RGB...")

    with rasterio.open(RASTER_S2) as src:

        if src.count < 4:
            print("Band Sentinel-2 tidak lengkap.")
            return

        transform, width, height, bounds = reprojection_grid(
            src,
            max_width=1700
        )

        # B04=Red, B03=Green, B02=Blue
        bands = [
            warper(src, i, transform, width, height, np.nan)
            for i in (3, 2, 1)
        ]

        rgb = np.stack(bands, axis=-1)

        valid = (
            np.isfinite(rgb).all(axis=2)
            & (rgb > 0).all(axis=2)
        )

        out = np.zeros(
            (height, width, 4),
            dtype=np.uint8
        )

        if valid.any():

            for j in range(3):

                channel = rgb[:, :, j]

                p_lo, p_hi = np.percentile(
                    channel[valid],
                    [2, 98]
                )

                stretched = np.clip(
                    (channel[valid] - p_lo)
                    / max(p_hi - p_lo, 1e-6),
                    0,
                    1
                )

                out[:, :, j][valid] = np.uint8(
                    np.round(stretched ** 0.9 * 255)
                )

            out[:, :, 3][valid] = 255

        folium.raster_layers.ImageOverlay(
            image=to_data_url(out),
            bounds=bounds,
            name="Citra Sentinel-2 lokal (100 m)",
            overlay=False,
            show=False,
            interactive=False
        ).add_to(peta)

    print("Basemap Sentinel-2 selesai.")


# ============================================================
# 6. MEMBACA PETA RANDOM FOREST 19 FITUR
# ============================================================

def main():

    print("=" * 65)
    print("PETA INTERAKTIF RF 19 FITUR TANPA MANGROVE TAMBAHAN")
    print("=" * 65)

    if not RASTER_RF.exists():
        raise FileNotFoundError(
            f"Raster koreksi RF 19 fitur tidak ditemukan:\n{RASTER_RF}\n"
            "Jalankan scripts/16_resampling_peta_50m.py dahulu."
        )

    with rasterio.open(RASTER_RF) as src:

        print("Raster:", RASTER_RF.name)
        print("Resolusi grid:", src.res)
        print("CRS:", src.crs)

        if src.count != 1:
            raise ValueError("Raster klasifikasi harus 1 band.")

        transform, width, height, bounds = reprojection_grid(
            src,
            max_width=2400
        )

        kelas_raster = np.zeros(
            (height, width),
            dtype=np.uint8
        )

        reproject(
            source=rasterio.band(src, 1),
            destination=kelas_raster,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata if src.nodata is not None else 0,
            dst_transform=transform,
            dst_crs="EPSG:4326",
            dst_nodata=0,
            resampling=Resampling.nearest
        )

    print("Raster klasifikasi berhasil dibaca.")


    # ========================================================
    # 7. MEMBUAT PETA FOLIUM
    # ========================================================

    peta = folium.Map(
        location=[-7.65, 112.9],
        tiles=None,
        zoom_start=8,
        control_scale=True,
        prefer_canvas=True
    )

    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Tiles © Esri",
        name="Citra satelit global (Esri)",
        overlay=False,
        control=True,
        show=True
    ).add_to(peta)

    make_satellite_true_color(peta)


    # ========================================================
    # 8. LAYER KLASIFIKASI
    # ========================================================

    print("\nMembuat layer klasifikasi...")

    for kode, (nama, warna, label) in KELAS.items():

        rgba = np.zeros(
            (height, width, 4),
            dtype=np.uint8
        )

        mask = kelas_raster == kode

        color = tuple(
            bytes.fromhex(warna.lstrip("#"))
        )

        rgba[mask, :3] = color
        rgba[mask, 3] = 190

        folium.raster_layers.ImageOverlay(
            image=to_data_url(rgba),
            bounds=bounds,
            name=f"Klasifikasi: {nama}",
            show=True,
            opacity=1.0,
            interactive=False,
            overlay=True,
            control=True
        ).add_to(peta)

        print(f"Layer {nama} selesai.")


    # ========================================================
    # 9. LAYER SAMPEL GEOJSON
    # ========================================================

    print("\nMembaca sampel GeoJSON...")

    geojson_files = sorted(
        GEOJSON_DIR.glob("*.geojson")
    )

    if len(geojson_files) == 1:

        with geojson_files[0].open(
            "r",
            encoding="utf-8"
        ) as f:
            geojson = json.load(f)

        features = geojson.get("features", [])

        print("Jumlah poligon:", len(features))

        for kode, (nama, warna, label) in KELAS.items():

            filtered = []

            for idx, ft in enumerate(features, 1):

                kelas = str(
                    (ft.get("properties") or {}).get("class", "")
                ).strip().lower()

                if kelas == label:

                    filtered.append({
                        "type": "Feature",
                        "geometry": ft.get("geometry"),
                        "properties": {
                            "class": label,
                            "id_sampel": idx
                        }
                    })

            if not filtered:
                continue

            group = folium.FeatureGroup(
                name=f"Sampel: {nama}",
                show=False,
                overlay=True
            )

            folium.GeoJson(
                {
                    "type": "FeatureCollection",
                    "features": filtered
                },
                style_function=lambda feature, c=warna: {
                    "color": c,
                    "weight": 2.5,
                    "opacity": 1,
                    "fillColor": c,
                    "fillOpacity": 0.14
                },
                highlight_function=lambda feature: {
                    "weight": 4,
                    "fillOpacity": 0.28
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=["class", "id_sampel"],
                    aliases=["Kelas:", "ID poligon:"]
                )
            ).add_to(group)

            group.add_to(peta)

            print(
                f"Sampel {nama}: {len(filtered)} poligon."
            )

    else:
        print(
            "PERHATIAN: Folder data/geojson harus "
            "berisi tepat satu file GeoJSON."
        )


    # ========================================================
    # 10. LEGENDA
    # ========================================================

    legends = "".join(
        (
            '<div style="display:flex;align-items:center;'
            'gap:8px;margin:5px 0">'
            f'<span style="background:{color};width:13px;'
            'height:13px;display:inline-block;'
            'border-radius:3px"></span>'
            f"<span>{name}</span></div>"
        )
        for _, (name, color, _) in KELAS.items()
    )

    legend_html = f"""
    <div style="
        position:fixed;
        bottom:26px;
        left:16px;
        z-index:9999;
        background:rgba(255,255,255,.96);
        color:#18202c;
        padding:13px 16px;
        box-shadow:0 4px 22px rgba(0,0,0,.22);
        border-radius:12px;
        font:13px Arial,sans-serif;
        min-width:155px;
    ">
        <strong style="font-size:14px">
            Legenda Tutupan Lahan
        </strong>

        {legends}

        <div style="
            margin-top:9px;
            padding-top:7px;
            border-top:1px solid #ddd;
            font-size:11px;
            color:#555;
        ">
            RF 19 fitur + koreksi DW (Mangrove asli RF)
        </div>
    </div>
    """

    peta.get_root().html.add_child(
        folium.Element(legend_html)
    )


    # ========================================================
    # 11. KONTROL LAYER
    # ========================================================

    folium.LayerControl(
        collapsed=False,
        position="topright",
        autoZIndex=False
    ).add_to(peta)

    peta.fit_bounds(bounds)


    # ========================================================
    # 12. MENYIMPAN HTML
    # ========================================================

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    peta.save(str(OUTPUT))

    print("\n" + "=" * 65)
    print("PETA INTERAKTIF TANPA MANGROVE TAMBAHAN SELESAI")
    print("=" * 65)

    print("Output:", OUTPUT)
    print("GeoJSON:", geojson_files[0].name if len(geojson_files) == 1 else "Tidak tersedia")
    print("Model: Random Forest 19 fitur")
    print("Resolusi data asli: 100 meter")
    print("Grid raster sumber peta: 50 meter")
    print("Overlay HTML maksimum: 2400 piksel")
    print("Peta lama tidak ditimpa.")
    print("PERHATIAN: Mangrove sesuai prediksi RF asli, tetapi koreksi Water Body menjadi Farm Land/Forest belum tervalidasi; bukan peta final.")


if __name__ == "__main__":
    main()
