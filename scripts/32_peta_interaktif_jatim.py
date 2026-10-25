"""TAHAP 32 — satu peta interaktif Jawa Timur dengan tile XYZ.

Input : outputs/maps/klasifikasi_jawa_timur_20m.tif
        data/geojson/batas_jawa_timur.geojson
Output: outputs/maps/peta_interaktif_jatim/index.html
        outputs/maps/peta_interaktif_jatim/tiles/{z}/{x}/{y}.png

Jalankan : py -3.9 scripts/32_peta_interaktif_jatim.py
Buka     : cd outputs/maps/peta_interaktif_jatim
           py -3.9 -m http.server 8000
           lalu http://localhost:8000

Hanya pustaka yang telah digunakan pada proyek: rasterio, numpy,
Pillow (PIL), geopandas, shapely. Tidak memakai GEE atau QGIS.
"""

from pathlib import Path
import argparse
import json
import math
import time

import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import from_bounds
from shapely.geometry import box
from shapely.prepared import prep

BASE = Path(__file__).resolve().parent.parent
INPUT = BASE / 'outputs/maps/klasifikasi_jawa_timur_20m.tif'
BORDER = BASE / 'data/geojson/batas_jawa_timur.geojson'
OUT = BASE / 'outputs/maps/peta_interaktif_jatim'

# 13 ~= 19 meter/piksel di sekitar lintang -7 derajat (setara detail asli).
DEFAULT_ZOOM_MIN = 7
DEFAULT_ZOOM_MAX = 13
TILE_SIZE = 256
HALF_WORLD = 20037508.342789244
WORLD = 2 * HALF_WORLD

# Kelas 0 transparan. Warna sesuai tahap klasifikasi dan mosaik.
PALETTE = np.array([
    [0, 0, 0, 0],        # NoData
    [230, 57, 70, 255],  # 1 Built-up
    [245, 196, 66, 255], # 2 Farm Land
    [48, 117, 211, 255], # 3 Water Body
    [45, 148, 81, 255],  # 4 Forest
    [155, 85, 182, 255], # 5 Mangrove
], dtype=np.uint8)


def tile_indices(bounds, z):
    """Semua XYZ tile yang melingkupi bounding box EPSG:3857."""
    xmin, ymin, xmax, ymax = bounds
    n = 2 ** z
    unit = WORLD / n
    eps = 1e-8
    x0 = max(0, min(n - 1, int(math.floor((xmin + HALF_WORLD) / unit))))
    x1 = max(0, min(n - 1, int(math.floor((xmax + HALF_WORLD - eps) / unit))))
    y0 = max(0, min(n - 1, int(math.floor((HALF_WORLD - ymax) / unit))))
    y1 = max(0, min(n - 1, int(math.floor((HALF_WORLD - ymin - eps) / unit))))
    return range(x0, x1 + 1), range(y0, y1 + 1)


def tile_bounds(z, x, y):
    unit = WORLD / (2 ** z)
    west = -HALF_WORLD + x * unit
    north = HALF_WORLD - y * unit
    return (west, north - unit, west + unit, north)


def make_html(geo, center, bounds, start_zoom, max_zoom):
    # Semua koordinat berurutan [lat, lng] di Leaflet.
    safe_geo = json.dumps(geo, ensure_ascii=False).replace('<', '\\u003c')
    safe_center = json.dumps(center)
    safe_bounds = json.dumps(bounds)
    legend = [
        ('#e63946', 'Built-up / Kawasan terbangun'),
        ('#f5c442', 'Farm Land / Lahan pertanian'),
        ('#3075d3', 'Water Body / Perairan'),
        ('#2d9451', 'Forest / Hutan'),
        ('#9b55b6', 'Mangrove'),
    ]
    rows = ''.join(f'<div class="legend-row"><i style="background:{color}"></i>{label}</div>'
                   for color, label in legend)
    return f'''<!DOCTYPE html>
<html lang="id">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Klasifikasi Tutupan Lahan Jawa Timur — Random Forest 20 m</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <style>
    html, body, #map {{ height:100%; margin:0; font-family:Arial,sans-serif; }}
    .heading {{ position:absolute; z-index:1000; left:12px; top:12px;
      background:rgba(255,255,255,.96); padding:12px 16px;
      box-shadow:0 2px 10px #0002; border-radius:8px; max-width:340px; }}
    .heading strong {{ display:block; font-size:15px; margin-bottom:5px; }}
    .heading small {{ color:#475569; line-height:1.4; }}
    .legend {{ background:#fff; padding:12px 14px; border-radius:8px;
      box-shadow:0 2px 10px #0002; font-size:12px; line-height:1.7; }}
    .legend strong {{display:block; margin-bottom:5px;}}
    .legend-row {{display:flex; align-items:center; gap:8px;}}
    .legend-row i {{ width:15px; height:15px; border-radius:3px; display:inline-block; flex-shrink:0;}}
    .leaflet-control-layers {{ font-size:13px; }}
    @media(max-width:650px) {{.heading {{max-width:240px; font-size:12px;}}}}
  </style>
</head>
<body>
  <div id="map"></div>
  <div class="heading"><strong>Klasifikasi Tutupan Lahan Jawa Timur</strong>
    <small>Sentinel-2 L2A • Random Forest • 5 kelas • raster asli 20 meter.<br>
    Data klasifikasi merupakan hasil model, bukan peta tutupan lahan resmi.</small></div>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const batas = {safe_geo};
    const peta = L.map('map', {{zoomControl:true, preferCanvas:true}});
    const petaDasar = L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      attribution:'&copy; OpenStreetMap contributors', maxZoom:18
    }}).addTo(peta);
    const klasifikasi = L.tileLayer('tiles/{{z}}/{{x}}/{{y}}.png', {{
      minZoom:{DEFAULT_ZOOM_MIN}, maxNativeZoom:{max_zoom}, maxZoom:18,
      opacity:0.87, noWrap:true, tms:false, attribution:'Klasifikasi RF Sentinel-2 20 m',
      errorTileUrl:''
    }}).addTo(peta);
    const batasLayer = L.geoJSON(batas, {{
      style:{{color:'#172554',weight:1.5,fill:false,opacity:0.9}}
    }}).addTo(peta);
    const luas = {safe_bounds};
    peta.fitBounds(luas, {{padding:[20,20]}});
    peta.setMaxZoom(18);
    L.control.layers({{'OpenStreetMap':petaDasar}},{{
      'Klasifikasi tutupan lahan':klasifikasi,
      'Batas Provinsi Jawa Timur':batasLayer
    }},{{collapsed:false}}).addTo(peta);
    const Legend = L.Control.extend({{
      options:{{position:'bottomright'}},
      onAdd:function(){{ const d=L.DomUtil.create('div','legend');
        d.innerHTML='<strong>Legenda lima kelas</strong>{rows}'; return d; }}
    }});
    peta.addControl(new Legend());
    L.control.scale({{imperial:false,metric:true}}).addTo(peta);
  </script>
</body>
</html>'''


def main():
    parser = argparse.ArgumentParser(description='Buat tile dan 1 peta HTML Jawa Timur')
    parser.add_argument('--min-zoom', type=int, default=DEFAULT_ZOOM_MIN)
    parser.add_argument('--max-zoom', type=int, default=DEFAULT_ZOOM_MAX)
    args = parser.parse_args()
    if not (0 <= args.min_zoom <= args.max_zoom <= 18):
        raise ValueError('Zoom tidak valid. Gunakan 0 <= min <= max <= 18.')
    if not INPUT.is_file() or not BORDER.is_file():
        raise FileNotFoundError(f'Periksa file input: {INPUT} dan {BORDER}')
    OUT.mkdir(parents=True, exist_ok=True)
    print('TAHAP 32 — PETA INTERAKTIF SELURUH JAWA TIMUR', flush=True)
    print('Input:', INPUT, flush=True)
    print('Output:', OUT, flush=True)

    gdf = gpd.read_file(BORDER)
    if gdf.empty or gdf.crs is None:
        raise ValueError('Batas Jawa Timur tidak valid / CRS kosong.')
    geographic = gdf.to_crs('EPSG:4326')
    geom_3857 = gdf.to_crs('EPSG:3857').geometry.union_all()
    prepared = prep(geom_3857)
    minlon, minlat, maxlon, maxlat = geographic.total_bounds
    bounds_ll = [[float(minlat),float(minlon)],[float(maxlat),float(maxlon)]]
    center = [(minlat + maxlat)/2, (minlon + maxlon)/2]
    geo = json.loads(geographic.to_json())

    with rasterio.open(INPUT) as src:
        if src.count != 1 or src.nodata != 0 or src.dtypes[0] != 'uint8':
            raise ValueError('Raster mosaik harus 1 band uint8 dengan NoData=0.')
        print('Raster:', src.width, 'x', src.height, src.crs, '|', src.res, flush=True)
        with WarpedVRT(src, crs='EPSG:3857', resampling=Resampling.nearest,
                       src_nodata=0, nodata=0, warp_mem_limit=128) as vrt:
            summary = {}
            total_created = 0
            for z in range(args.min_zoom, args.max_zoom+1):
                t0 = time.time()
                xs, ys = tile_indices(geom_3857.bounds, z)
                candidate = 0
                created = 0
                skipped = 0
                for x in xs:
                    directory = OUT / 'tiles' / str(z) / str(x)
                    for y in ys:
                        bb = tile_bounds(z,x,y)
                        if not prepared.intersects(box(*bb)):
                            continue
                        candidate += 1
                        path = directory / f'{y}.png'
                        if path.exists() and path.stat().st_size > 0:
                            skipped += 1
                            continue
                        win = from_bounds(*bb, transform=vrt.transform)
                        # Lebih efisien dibanding memuat raster 31013x22016 ke RAM.
                        pixels = vrt.read(1, window=win,
                                          out_shape=(TILE_SIZE,TILE_SIZE),
                                          resampling=Resampling.nearest,
                                          boundless=False)
                        if not np.any(pixels):
                            continue
                        if np.any(pixels > 5):
                            raise ValueError(f'Nilai kelas di luar 0..5 pada tile {z}/{x}/{y}')
                        rgba = PALETTE[pixels]
                        directory.mkdir(parents=True, exist_ok=True)
                        tmp_path = path.with_suffix('.tmp.png')
                        Image.fromarray(rgba, 'RGBA').save(tmp_path, optimize=True)
                        tmp_path.replace(path)
                        created += 1
                        total_created += 1
                summary[str(z)] = {'cakupan_tile':candidate,'tile_baru':created,'tile_cache':skipped}
                print(f'[ZOOM {z}] kandidat={candidate:,} | baru={created:,} | '
                      f'cache={skipped:,} | {time.time()-t0:.1f} detik',flush=True)

    html = make_html(geo, center, bounds_ll, args.min_zoom, args.max_zoom)
    # Karena --min-zoom bisa diganti, samakan setting layer dengan batas aktual.
    html = html.replace(f'minZoom:{DEFAULT_ZOOM_MIN},', f'minZoom:{args.min_zoom},')
    (OUT/'index.html').write_text(html,encoding='utf-8')
    (OUT/'info_tiles.json').write_text(json.dumps({
       'sumber':str(INPUT),'resolusi_sumber_meter':20,
       'min_zoom':args.min_zoom,'max_zoom':args.max_zoom,
       'tile':summary,'catatan':'Tile PNG mempertahankan label kelas menggunakan nearest-neighbor.'
    },indent=2,ensure_ascii=False),encoding='utf-8')
    print('\nSELESAI! Peta:',OUT/'index.html',flush=True)
    print('Tile PNG baru:',total_created,flush=True)
    print('Untuk membuka peta:',flush=True)
    print('  cd outputs/maps/peta_interaktif_jatim',flush=True)
    print('  py -3.9 -m http.server 8000',flush=True)
    print('Lalu buka: http://localhost:8000',flush=True)
    print('Catatan: pustaka Leaflet dan peta dasar OSM membutuhkan internet.',flush=True)

if __name__ == '__main__':
    main()
