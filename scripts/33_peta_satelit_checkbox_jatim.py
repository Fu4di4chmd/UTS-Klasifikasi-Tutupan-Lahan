"""Tahap 33: Peta Jawa Timur berlatar Esri dengan checkbox per kelas.

Prasyarat: Tahap 32 telah menghasilkan outputs/maps/peta_interaktif_jatim/tiles.
Tidak mengambil Sentinel-2 lagi. Membagi tile klasifikasi yang sudah ada
menjadi lima lapisan PNG transparan, kemudian membuat HTML Leaflet.

Jalankan dari root proyek:
    py -3.9 scripts/33_peta_satelit_checkbox_jatim.py

Buka menggunakan server lokal pada folder peta_interaktif_jatim:
    cd outputs/maps/peta_interaktif_jatim
    py -3.9 -m http.server 8000
    http://localhost:8000/index_kelas.html
"""

from pathlib import Path
import json
import numpy as np
from PIL import Image
import geopandas as gpd

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "outputs" / "maps" / "peta_interaktif_jatim"
TILES = OUT / "tiles"
CLASS_TILES = OUT / "kelas_tiles"
BORDER = BASE / "data" / "geojson" / "batas_jawa_timur.geojson"
HTML = OUT / "index_kelas.html"

CLASSES = [
    (1, "builtup", "Built-up / Kawasan terbangun", "#e63946", (230, 57, 70)),
    (2, "farmland", "Farm Land / Lahan pertanian", "#f5c442", (245, 196, 66)),
    (3, "water", "Water Body / Perairan", "#3075d3", (48, 117, 211)),
    (4, "forest", "Forest / Hutan", "#2d9451", (45, 148, 81)),
    (5, "mangrove", "Mangrove", "#9b55b6", (155, 85, 182)),
]


def main():
    if not TILES.is_dir() or not BORDER.is_file():
        raise FileNotFoundError(
            "Jalankan Tahap 32 dahulu dan pastikan batas_jawa_timur.geojson tersedia."
        )
    sources = sorted(TILES.glob("*/*/*.png"))
    if not sources:
        raise FileNotFoundError("Tidak ada tile hasil Tahap 32 dalam folder tiles.")

    print("TAHAP 33 — SATELIT ESRI + CHECKBOX 5 KELAS", flush=True)
    print(f"Tile sumber: {len(sources):,}", flush=True)
    made, cached = 0, 0
    # Bagi gambar asli sesuai warna eksak; RGBA kelas lain menjadi transparan.
    # Ini menghindari membaca ulang GeoTIFF raksasa dan tetap memakai label Tahap 32.
    for idx, src in enumerate(sources, 1):
        with Image.open(src) as im:
            rgba = np.asarray(im.convert("RGBA"), dtype=np.uint8)
        visible = rgba[:, :, 3] > 0
        for _, slug, _, _, rgb in CLASSES:
            dest = CLASS_TILES / slug / src.relative_to(TILES)
            if dest.is_file() and dest.stat().st_size > 0:
                cached += 1
                continue
            is_class = visible & np.all(rgba[:, :, :3] == rgb, axis=2)
            # Tile transparan tetap dibuat agar permintaan peta tidak menghasilkan 404.
            result = np.zeros_like(rgba)
            if np.any(is_class):
                result[is_class, :3] = rgb
                result[is_class, 3] = rgba[is_class, 3]
            dest.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(result).save(dest, format="PNG", compress_level=1)
            made += 1
        if idx % 300 == 0 or idx == len(sources):
            print(f"[TILE] {idx:,}/{len(sources):,} | baru={made:,} | cache={cached:,}", flush=True)

    gdf = gpd.read_file(BORDER)
    if gdf.empty or gdf.crs is None:
        raise ValueError("GeoJSON batas provinsi kosong atau tanpa CRS")
    gdf = gdf.to_crs("EPSG:4326")
    minlon, minlat, maxlon, maxlat = map(float, gdf.total_bounds)
    bounds = [[minlat, minlon], [maxlat, maxlon]]
    batas_js = json.dumps(json.loads(gdf.to_json()), ensure_ascii=False).replace("<", "\\u003c")
    bounds_js = json.dumps(bounds)
    legend = "\n".join(
        f'<div class="item"><span class="chip" style="background:{color}"></span>{label}</div>'
        for _, _, label, color, _ in CLASSES
    )
    overlay_lines = "\n".join(
        f'    const layer_{slug} = L.tileLayer("kelas_tiles/{slug}/{{z}}/{{x}}/{{y}}.png", '
        f'{{minZoom:7, maxNativeZoom:13, maxZoom:18, opacity:0.90, noWrap:true}}).addTo(map);'
        for _, slug, _, _, _ in CLASSES
    )
    layer_map = ",\n".join(
        f'      "{label}": layer_{slug}' for _, slug, label, _, _ in CLASSES
    )
    html = '''<!doctype html>
<html lang="id">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Klasifikasi Tutupan Lahan Jawa Timur | 20 m</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>
html,body,#map {height:100%;margin:0;font-family:Arial,Helvetica,sans-serif}
.heading {position:absolute;z-index:999;top:14px;left:58px;background:rgba(255,255,255,.95);border-radius:9px;padding:12px 16px;max-width:340px;box-shadow:0 2px 10px #0002}
.heading strong {font-size:16px;display:block;margin-bottom:5px}
.heading small {color:#475569;line-height:1.45}
.legend {background:rgba(255,255,255,.96);padding:13px 16px;border-radius:9px;box-shadow:0 2px 10px #0003;line-height:1.65;font-size:13px}
.legend strong {display:block;margin-bottom:6px}
.item {display:flex;align-items:center;gap:9px}
.chip {width:16px;height:16px;border-radius:3px;display:inline-block;flex-shrink:0}
.leaflet-control-layers {font-size:13px;line-height:1.7}
.leaflet-control-layers label {cursor:pointer}
@media(max-width:650px){.heading{max-width:220px;font-size:12px;padding:8px 10px}.legend{font-size:11px;padding:8px 10px}}
</style>
</head>
<body>
<div id="map"></div>
<div class="heading"><strong>Klasifikasi Tutupan Lahan Jawa Timur</strong>
<small>Sentinel-2 L2A • Random Forest • 5 kelas • raster sumber 20 meter.<br>
Hasil prediksi model, bukan peta tutupan lahan resmi.</small></div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const map = L.map("map", {preferCanvas:true, zoomControl:true});
const satellite = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
  maxZoom:18, attribution:'Tiles &copy; Esri, Maxar, Earthstar Geographics, and the GIS User Community'
}).addTo(map);
const osm = L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom:18, attribution:'&copy; OpenStreetMap contributors'
});
__OVERLAYS__
const batas = __GEOJSON__;
const boundary = L.geoJSON(batas, {style:{color:'#263f8d',weight:1.7,fill:false,opacity:0.95}});
const overlays = {
__LAYER_MAP__,
      "Batas Provinsi Jawa Timur": boundary
};
L.control.layers({"Citra satelit Esri":satellite,"OpenStreetMap":osm}, overlays,
  {collapsed:false,position:'topright'}).addTo(map);
map.fitBounds(__BOUNDS__, {padding:[18,18]});
const Legend = L.Control.extend({options:{position:'bottomleft'},onAdd:function(){
 const d=L.DomUtil.create('div','legend');
 d.innerHTML='<strong>Legenda Tutupan Lahan</strong>__LEGEND__';
 return d;
}});
map.addControl(new Legend());
L.control.scale({imperial:false,metric:true}).addTo(map);
</script>
</body></html>
'''
    html = (html.replace("__OVERLAYS__", overlay_lines)
                .replace("__GEOJSON__", batas_js)
                .replace("__BOUNDS__", bounds_js)
                .replace("__LAYER_MAP__", layer_map)
                .replace("__LEGEND__", json.dumps(legend)))
    HTML.write_text(html, encoding="utf-8")
    print("\nSELESAI!", flush=True)
    print("Peta:", HTML, flush=True)
    print("Tile kelas baru:", made, "| cache:", cached, flush=True)
    print("Jalankan server dari folder outputs/maps/peta_interaktif_jatim:", flush=True)
    print("  py -3.9 -m http.server 8000", flush=True)
    print("Buka http://localhost:8000/index_kelas.html", flush=True)


if __name__ == "__main__":
    main()
