"""Audit kandidat piksel Mangrove menggunakan all_touched pada raster 100 m.

Jalankan dari root proyek:
    py -3.9 scripts/02_uji_all_touched_mangrove.py

Tidak mengubah GeoJSON, data train/test, ataupun model.
Output: outputs/reports/audit_mangrove_all_touched/

Penting: all_touched mengidentifikasi piksel yang BERSINGGUNGAN, tidak menjamin
piksel tersebut berisi Mangrove. Kandidat perlu divalidasi pada citra sebelum
menjadi label training.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from shapely.geometry import box
from shapely.strtree import STRtree

BASE = Path(__file__).resolve().parent.parent
GEOJSON_DIR = BASE / 'data' / 'geojson'
RASTER = BASE / 'data' / 'prediksi' / 'sentinel2_jatim_training_buffer3km_14fitur_100m.tif'
OUT = BASE / 'outputs' / 'reports' / 'audit_mangrove_all_touched'
OUT.mkdir(parents=True, exist_ok=True)
FEATURES = ['B02','B03','B04','B05','B06','B07','B08','B8A','B11','B12','NDVI','NDMI','MNDWI','NDBI']


def main():
    paths = sorted(GEOJSON_DIR.glob('*.geojson'))
    if len(paths) != 1:
        raise ValueError(f'Folder {GEOJSON_DIR} harus berisi tepat satu GeoJSON; ditemukan {len(paths)}')
    if not RASTER.is_file():
        raise FileNotFoundError(RASTER)
    gdf = gpd.read_file(paths[0])
    if gdf.crs is None or 'class' not in gdf.columns:
        raise ValueError('GeoJSON perlu CRS dan kolom class')
    gdf['class'] = gdf['class'].astype(str).str.strip().str.lower()
    gdf['polygon_id'] = np.arange(1, len(gdf)+1)
    mang = gdf.loc[gdf['class'] == 'mangrove'].copy()
    other = gdf.loc[gdf['class'] != 'mangrove'].copy()
    if mang.empty:
        raise ValueError('Tidak ada poligon mangrove')
    if not gdf.geometry.notna().all() or not gdf.is_valid.all():
        raise ValueError('GeoJSON berisi geometri kosong/tidak valid; perbaiki dahulu')

    print('GeoJSON:', paths[0].name, '| total:', len(gdf), '| Mangrove:', len(mang))
    with rasterio.open(RASTER) as src:
        if src.count != 14 or list(src.descriptions) != FEATURES:
            raise ValueError('Raster 14 fitur memiliki jumlah/urutan band yang tidak sesuai')
        mang = mang.to_crs(src.crs)
        other = other.to_crs(src.crs)
        transform = src.transform
        # Baca hanya satu band untuk syarat minimum dan identifikasi candidates;
        # validitas 14 fitur dicek hanya untuk kandidat unik di bawah.
        shape = (src.height, src.width)
        def burn(df, touched):
            return rasterize(((geom, int(pid)) for geom, pid in zip(df.geometry, df.polygon_id)),
                             out_shape=shape, transform=transform, fill=0,
                             dtype='int32', all_touched=touched)
        center_m = burn(mang, False)
        touch_m = burn(mang, True)
        # Konflik antarkelas konservatif: bila suatu piksel bahkan BERSINGGUNGAN
        # dengan kelas lain, jangan usulkan sebagai sampel Mangrove baru.
        touch_other = burn(other, True)
        candidates = (touch_m > 0) & (center_m == 0)
        conflict = candidates & (touch_other > 0)
        candidates_safe = candidates & (touch_other == 0)
        rr, cc = np.where(candidates_safe)
        print('Piksel pusat Mangrove:', int(np.count_nonzero(center_m)))
        print('Kandidat tambahan all_touched:', int(np.count_nonzero(candidates)))
        print('Konflik dengan kelas lain (dikeluarkan):', int(np.count_nonzero(conflict)))
        print('Kandidat tanpa konflik:', len(rr))

        if len(rr):
            # Hindari membaca keseluruhan data 14 band ke RAM sekaligus.
            ok = np.ones(len(rr), dtype=bool)
            for i, feature in enumerate(FEATURES, 1):
                band = src.read(i, masked=True)
                arr = np.asarray(band[rr, cc].filled(np.nan), dtype=np.float32)
                ok &= np.isfinite(arr)
                if i <= 10:
                    ok &= arr > 0
                print(f'Validasi {i:02d}/14: {feature}')
                del band
            rr, cc = rr[ok], cc[ok]
            print('Kandidat tambahan valid 14 fitur:', len(rr))
        else:
            print('Tidak ada kandidat tanpa konflik; CSV detail tetap dibuat.')

        # Mengukur irisan geometri/piksel (akurasi spasial, bukan kemurnian kelas).
        # Untuk kandidat yang bertumpuk dengan >1 poligon Mangrove, gunakan
        # poligon dengan luas irisan terbesar; konflik antarmangrove dicatat.
        geoms = list(mang.geometry)
        ids = mang['polygon_id'].to_numpy(dtype=int)
        tree = STRtree(geoms)
        idx_by_wkb = {}
        for j, geom in enumerate(geoms):
            idx_by_wkb.setdefault(geom.wkb, []).append(j)
        rows = []
        cell_area = abs(transform.a * transform.e)
        for r, c in zip(rr, cc):
            x0, y0 = transform * (int(c), int(r))
            pix = box(x0, y0 + transform.e, x0 + transform.a, y0)
            matches = tree.query(pix)
            matches_indices = []
            for match in matches:
                if isinstance(match, (int, np.integer)):
                    matches_indices.append(int(match))
                else:  # Shapely 1.x returns geom objects
                    matches_indices.extend(idx_by_wkb.get(match.wkb, []))
            scores = []
            for j in set(matches_indices):
                inter = geoms[j].intersection(pix).area
                if inter > 0:
                    scores.append((inter, int(ids[j])))
            if not scores:
                continue
            scores.sort(reverse=True)
            xcent, ycent = rasterio.transform.xy(transform, int(r), int(c), offset='center')
            rows.append({
                'polygon_id_usulan': scores[0][1], 'row': int(r), 'col': int(c),
                'x_utm': xcent, 'y_utm': ycent,
                'fraction_area_polygon': round(scores[0][0] / cell_area, 5),
                'jumlah_poligon_mangrove_menyentuh': len(scores),
                'status': 'PERLU VALIDASI VISUAL; BELUM BOLEH DIPAKAI TRAINING'
            })

        df = pd.DataFrame(rows, columns=['polygon_id_usulan','row','col','x_utm','y_utm',
                                         'fraction_area_polygon', 'jumlah_poligon_mangrove_menyentuh','status'])
        df.to_csv(OUT / 'kandidat_tambahan_all_touched.csv', index=False)
        thresholds = [0.1, 0.25, 0.5, 0.75]
        summaries = []
        for threshold in thresholds:
            num = int((df.fraction_area_polygon >= threshold).sum()) if len(df) else 0
            summaries.append({'ambang_luas_irisan_min': threshold, 'kandidat': num})
        pd.DataFrame(summaries).to_csv(OUT / 'ringkasan_ambang_luas.csv', index=False)
        print('\nKandidat menurut proporsi piksel yang tercakup poligon:')
        print(pd.DataFrame(summaries).to_string(index=False))
        print('CSV detail:', OUT / 'kandidat_tambahan_all_touched.csv')
        print('CSV ringkasan:', OUT / 'ringkasan_ambang_luas.csv')
        print('BELUM mengubah data training/testing; perlu validasi citra untuk kandidat piksel tepi.')

if __name__ == '__main__':
    main()
