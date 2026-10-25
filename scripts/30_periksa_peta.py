
from pathlib import Path
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

BASE = Path(__file__).resolve().parents[1]
FOLDER = BASE / "outputs/maps/klasifikasi_20m"
OUT = FOLDER / "preview"
OUT.mkdir(parents=True, exist_ok=True)

BLOK = [
    "B_32_459",
    "B_34_459",
    "B_34_460",
    "B_28_459",
    "B_28_456"
]

NAMA = [
    "Built-up", "Farm Land", "Water Body",
    "Forest", "Mangrove"
]

WARNA = [
    "#DD3333",
    "#F2C441",
    "#2374E1",
    "#238741",
    "#8737A5"
]

cmap = ListedColormap(WARNA)
cmap.set_bad("#E5E7EB")
norm = BoundaryNorm(np.arange(0.5, 6.5, 1), cmap.N)

legenda = [
    Patch(facecolor=w, label=n)
    for w, n in zip(WARNA, NAMA)
]
legenda.append(Patch(facecolor="#E5E7EB", label="NoData"))

for blok in BLOK:
    path = FOLDER / f"klasifikasi_rf_{blok}_20m.tif"

    if not path.exists():
        raise FileNotFoundError(path)

    with rasterio.open(path) as src:
        raster = src.read(1)
        bounds = src.bounds

    raster = np.ma.masked_equal(raster, 0)

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(
        raster,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        extent=[
            bounds.left, bounds.right,
            bounds.bottom, bounds.top
        ]
    )

    ax.set_title(
        f"Klasifikasi Tutupan Lahan 20 m\n{blok}",
        fontsize=14
    )
    ax.set_xlabel("Easting UTM (meter)")
    ax.set_ylabel("Northing UTM (meter)")

    ax.legend(
        handles=legenda,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        frameon=False
    )

    ax.set_aspect("equal")

    output = OUT / f"preview_{blok}.png"

    fig.savefig(
        output,
        dpi=160,
        bbox_inches="tight"
    )
    plt.close(fig)

    print(f"Berhasil: {output}")

print("\nSEMUA PREVIEW PETA SELESAI")
print("Folder:", OUT)
