from pathlib import Path
import os
import warnings

import cv2
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from tqdm import tqdm

# ===================== KONFIGURACJA =====================

# ZMIANA: automatyczna baza ścieżek względem lokalizacji tego pliku
HERE = Path(__file__).resolve().parent  # ZMIANA

# Jeśli dane masz obok pliku:
#   projekt/
#     preprocess_spacenet_paris.py
#     SN2_buildings_train_AOI_3_Paris/
# to nic nie zmieniaj poniżej.
BASE_DIR = (HERE / "SN2_buildings_train_AOI_3_Paris/AOI_3_Paris_Train").resolve()  # ZMIANA
OUT_DIR  = (HERE / "dataset_paris_small").resolve()              # ZMIANA

# ile obrazków przerobić (None = wszystkie). Na start 200–300.
LIMIT_IMAGES = 300

# opcjonalnie zmniejsz obraz/maskę do kwadratu NxN (None = bez zmiany)
RESIZE_TO = 256

# czy nadpisywać istniejące pliki wynikowe
OVERWRITE = False

# ========================================================

IMG_DIR = (BASE_DIR / "RGB-PanSharpen")        # ZMIANA
GEOJSON_DIR = (BASE_DIR / "geojson" / "buildings")  # ZMIANA

OUT_IMAGES = (OUT_DIR / "images")              # ZMIANA
OUT_MASKS  = (OUT_DIR / "masks")               # ZMIANA

OUT_IMAGES.mkdir(parents=True, exist_ok=True)
OUT_MASKS.mkdir(parents=True, exist_ok=True)

def stretch_to_uint8(img: np.ndarray) -> np.ndarray:
    """Proste rozciąganie histogramu (2–98 pctl) per kanał -> uint8."""
    img = img.astype(np.float32)
    out = np.zeros_like(img, dtype=np.uint8)
    for c in range(img.shape[2]):
        ch = img[:, :, c]
        lo = np.percentile(ch, 2)
        hi = np.percentile(ch, 98)
        if hi <= lo:
            hi = ch.max()
            lo = ch.min()
        ch = np.clip((ch - lo) / (hi - lo + 1e-6), 0, 1) * 255.0
        out[:, :, c] = ch.astype(np.uint8)
    return out

def load_image_rgb(path_tif: Path) -> np.ndarray:  # ZMIANA: Path zamiast str
    with rasterio.open(path_tif.as_posix()) as src:
        bands = [1, 2, 3] if src.count >= 3 else list(range(1, src.count + 1))
        img = src.read(bands)                     # (C,H,W)
        img = np.transpose(img, (1, 2, 0))        # -> (H,W,C)
        if img.dtype != np.uint8:
            img = stretch_to_uint8(img)
        return img

def rasterize_buildings_mask(geojson_path: Path, ref_tif_path: Path) -> np.ndarray:  # ZMIANA
    with rasterio.open(ref_tif_path.as_posix()) as src:
        out_shape = (src.height, src.width)
        transform = src.transform
        img_crs   = src.crs

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gdf = gpd.read_file(geojson_path)  # Path działa w GeoPandas

    if gdf.empty:
        return np.zeros(out_shape, dtype=np.uint8)

    try:
        if gdf.crs is not None and img_crs is not None and gdf.crs != img_crs:
            gdf = gdf.to_crs(img_crs)
    except Exception:
        pass

    shapes = [(geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty]
    if not shapes:
        return np.zeros(out_shape, dtype=np.uint8)

    mask = rasterize(
        shapes=shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        all_touched=True,
        dtype=np.uint8
    )
    return mask

def maybe_resize(img: np.ndarray, mask: np.ndarray, size: int | None):
    if size is None:
        return img, mask
    img_res  = cv2.resize(img,  (size, size), interpolation=cv2.INTER_AREA)
    mask_res = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return img_res, mask_res

def main():
    tif_files = sorted([p for p in IMG_DIR.glob("*.tif")])  # ZMIANA
    geojson_stems = {p.stem for p in GEOJSON_DIR.glob("*.geojson")}  # ZMIANA

    processed = 0
    skipped_missing_geo = 0
    skipped_missing_img  = 0

    pbar_iter = tif_files[:LIMIT_IMAGES] if LIMIT_IMAGES is not None else tif_files

    for tif_path in tqdm(pbar_iter, desc="Przetwarzanie"):
        stem = tif_path.stem
        tif_id = stem.replace("RGB-PanSharpen_", "")
        geo_path = GEOJSON_DIR / f"buildings_{tif_id}.geojson"

        if not geo_path.exists():
            skipped_missing_geo += 1
            continue

        out_png = f"{stem}.png"
        out_img_path = OUT_IMAGES / out_png
        out_msk_path = OUT_MASKS  / out_png

        if not OVERWRITE and out_img_path.exists() and out_msk_path.exists():
            processed += 1
            continue

        if not tif_path.exists():
            skipped_missing_img += 1
            continue

        try:
            img = load_image_rgb(tif_path)                              # (H,W,3) uint8
            mask01 = rasterize_buildings_mask(geo_path, tif_path)       # (H,W)   0/1
            img, mask01 = maybe_resize(img, mask01, RESIZE_TO)

            cv2.imwrite(out_img_path.as_posix(), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))  # ZMIANA
            cv2.imwrite(out_msk_path.as_posix(), (mask01.astype(np.uint8) * 255))       # ZMIANA
            processed += 1
        except Exception as e:
            print(f"[WARN] Problem z {tif_path.name}: {e}")

    print("\n=== PODSUMOWANIE ===")
    print(f"Zapisanych par obraz+mask: {processed}")
    print(f"Pominięte (brak geojson): {skipped_missing_geo}")
    print(f"Pominięte (brak tif):     {skipped_missing_img}")
    print(f"Wyjście: {OUT_DIR}")

if __name__ == "__main__":
    main()
