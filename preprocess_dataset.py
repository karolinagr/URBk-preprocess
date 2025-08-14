from pathlib import Path
import re
import json
import warnings
import glob

import numpy as np
import cv2
import geopandas as gpd
import fiona
from shapely.geometry import box
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_bounds
from tqdm import tqdm

# ====================== CONFIG ======================
HERE = Path(__file__).resolve().parent

# --- Data roots (adjust to your disk layout) ---
# SN2: images + buildings
BASE_SN2 = HERE / "SN2_buildings_train_AOI_3_Paris" / "AOI_3_Paris_Train"
IMG_DIR_SN2 = BASE_SN2 / "RGB-PanSharpen"
B_DIR_SN2   = BASE_SN2 / "geojson" / "buildings"

# SN3: roads
BASE_SN3 = HERE / "SN3_roads_train_AOI_3_Paris" / "AOI_3_Paris"
R_DIR_SN3 = BASE_SN3 / "geojson_roads"

# --- Output ---
OUT_DIR      = (HERE / "dataset_paris_patches").resolve()
OUT_IMAGES   = OUT_DIR / "images"
OUT_MASKS    = OUT_DIR / "masks"
OUT_VIZ      = OUT_DIR / "viz"          # optional colored previews
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_IMAGES.mkdir(parents=True, exist_ok=True)
OUT_MASKS.mkdir(parents=True, exist_ok=True)
OUT_VIZ.mkdir(parents=True, exist_ok=True)

# --- Patch settings (the "big effect" part) ---
PATCH_SIZE       = 256      # final patch size
STRIDE           = 128      # overlap to increase diversity
MIN_ROAD_PX      = 200      # keep a patch only if it has >= this many road pixels
KEEP_BG_PROB     = 0.10     # probability to also keep a road-free patch (for stability)
RESIZE_PATCH_TO  = None     # set to e.g. 256 to force resize; None keeps native 256×256

# --- Roads rasterization ---
ROAD_BUFFER_METERS = 5.0    # wider roads -> easier learning (increase from 3.0)
ALL_TOUCHED        = True

# --- Classes: single-channel mask with indices
CLS_BG, CLS_ROAD, CLS_BLDG = 0, 1, 2
CLASS_OVERLAY_ORDER = ["road", "building"]  # later entry wins on overlap

# --- Overwrite behavior ---
OVERWRITE = False           # set True to regenerate patches

# =====================================================


# ----------------- HELPERS -----------------
def stretch_to_uint8(img: np.ndarray) -> np.ndarray:
    """Simple 2–98 percentile per-channel stretch to uint8."""
    img = img.astype(np.float32)
    out = np.zeros_like(img, dtype=np.uint8)
    for c in range(img.shape[2]):
        ch = img[:, :, c]
        lo, hi = np.percentile(ch, 2), np.percentile(ch, 98)
        if hi <= lo:
            lo, hi = ch.min(), ch.max()
        ch = np.clip((ch - lo) / (hi - lo + 1e-6), 0, 1) * 255.0
        out[:, :, c] = ch.astype(np.uint8)
    return out

def load_image_rgb(path_tif: Path):
    """Read RGB and raster meta (transform, crs, width, height, bounds)."""
    with rasterio.open(path_tif.as_posix()) as src:
        meta = dict(
            transform=src.transform,
            crs=src.crs,
            width=src.width,
            height=src.height,
            bounds=src.bounds,
        )
        bands = [1, 2, 3] if src.count >= 3 else list(range(1, src.count + 1))
        img = np.transpose(src.read(bands), (1, 2, 0))
        if img.dtype != np.uint8:
            img = stretch_to_uint8(img)
        return img, meta

def buildings_geo_for_image(tif_path: Path) -> Path | None:
    """
    SN2 naming:
      image:    RGB-PanSharpen_AOI_3_Paris_imgXYZ.tif
      buildings: buildings_AOI_3_Paris_imgXYZ.geojson
    """
    stem = tif_path.stem
    tif_id = re.sub(r"^RGB[-_]?PanSharpen[_-]?", "", stem, flags=re.I)  # AOI_3_Paris_imgXYZ
    cand = B_DIR_SN2 / f"buildings_{tif_id}.geojson"
    return cand if cand.exists() else None

def rasterize_buildings(geo_path: Path, out_shape, transform, class_value=CLS_BLDG):
    """Rasterize building polygons to class_value; return zeros if missing."""
    if geo_path is None or not geo_path.exists():
        return np.zeros(out_shape, dtype=np.uint8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gdf = gpd.read_file(geo_path)
    if gdf.empty:
        return np.zeros(out_shape, dtype=np.uint8)
    shapes = [(geom, class_value) for geom in gdf.geometry if geom and not geom.is_empty]
    if not shapes:
        return np.zeros(out_shape, dtype=np.uint8)
    m = rasterize(shapes=shapes, out_shape=out_shape, transform=transform,
                  fill=0, all_touched=ALL_TOUCHED, dtype=np.uint8)
    m[m > 0] = class_value
    return m

def index_roads_bboxes(roads_root: Path) -> gpd.GeoDataFrame:
    """
    Build a GeoDataFrame index of SN3 road files using only file-level bounding boxes.
    Fast: reads bounds via fiona, not full geometry. Stored in EPSG:4326.
    """
    paths = list(roads_root.rglob("*.geojson"))
    recs = []
    for p in paths:
        try:
            with fiona.open(p.as_posix()) as src:
                b = src.bounds  # (minx,miny,maxx,maxy), commonly in 4326
                crs = src.crs or {"init": "epsg:4326"}
            recs.append({"path": str(p), "geometry": box(*b), "src_crs": crs})
        except Exception as e:
            print(f"[WARN] Could not read bounds for {p.name}: {e}")
    if not recs:
        return gpd.GeoDataFrame(columns=["path", "geometry", "src_crs"], geometry="geometry", crs="EPSG:4326")
    gdf = gpd.GeoDataFrame(recs, geometry="geometry", crs="EPSG:4326")
    try:
        gdf = gdf.to_crs(4326)
    except Exception:
        pass
    _ = gdf.sindex
    return gdf

def roads_mask_for_image_bbox(img_path: Path, roads_index_gdf: gpd.GeoDataFrame,
                              out_shape, transform, img_crs,
                              road_buffer_m=ROAD_BUFFER_METERS) -> np.ndarray:
    """
    1) Compute image bbox in EPSG:4326
    2) Query index for road files whose bbox intersects
    3) Read those, reproject to image CRS, buffer lines, rasterize to class 1
    """
    if roads_index_gdf.empty:
        return np.zeros(out_shape, dtype=np.uint8)

    with rasterio.open(img_path.as_posix()) as src:
        minx, miny, maxx, maxy = transform_bounds(img_crs, "EPSG:4326",
                                                  *src.bounds, densify_pts=21)
    bbox4326 = box(minx, miny, maxx, maxy)

    cand_idx = list(roads_index_gdf.sindex.intersection(bbox4326.bounds))
    if not cand_idx:
        return np.zeros(out_shape, dtype=np.uint8)
    candidates = roads_index_gdf.iloc[cand_idx]
    candidates = candidates[candidates.intersects(bbox4326)]
    if candidates.empty:
        return np.zeros(out_shape, dtype=np.uint8)

    parts = []
    for _, row in candidates.iterrows():
        p = Path(row["path"])
        try:
            g = gpd.read_file(p.as_posix())
            if g.empty:
                continue
            if g.crs is None:
                g = g.set_crs(4326)
            else:
                try:
                    g = g.to_crs(4326)
                except Exception:
                    pass
            parts.append(g)
        except Exception as e:
            print(f"[WARN] Read failed for {p.name}: {e}")
    if not parts:
        return np.zeros(out_shape, dtype=np.uint8)

    roads = gpd.pd.concat(parts, ignore_index=True)
    if roads.empty:
        return np.zeros(out_shape, dtype=np.uint8)

    try:
        roads = roads.to_crs(img_crs)
    except Exception:
        return np.zeros(out_shape, dtype=np.uint8)

    try:
        if img_crs and getattr(img_crs, "is_projected", False):
            roads["geometry"] = roads.buffer(road_buffer_m)
        else:
            roads_m = roads.to_crs(3857).buffer(road_buffer_m)
            roads = gpd.GeoDataFrame(geometry=roads_m, crs=3857).to_crs(img_crs)
    except Exception:
        pass

    shapes = [(geom, CLS_ROAD) for geom in roads.geometry if geom and not geom.is_empty]
    if not shapes:
        return np.zeros(out_shape, dtype=np.uint8)

    m = rasterize(
        shapes=shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        all_touched=ALL_TOUCHED,
        dtype=np.uint8
    )
    m[m > 0] = CLS_ROAD
    return m

def save_patch(img_patch: np.ndarray, mask_patch: np.ndarray, base_name: str, y: int, x: int):
    """Save image/mask (and optional viz) for a extracted patch."""
    name = f"{base_name}_{y}_{x}.png"
    ip = img_patch
    mp = mask_patch

    # optional resize to a fixed size
    if RESIZE_PATCH_TO is not None and RESIZE_PATCH_TO != PATCH_SIZE:
        ip = cv2.resize(ip, (RESIZE_PATCH_TO, RESIZE_PATCH_TO), interpolation=cv2.INTER_AREA)
        mp = cv2.resize(mp, (RESIZE_PATCH_TO, RESIZE_PATCH_TO), interpolation=cv2.INTER_NEAREST)

    cv2.imwrite((OUT_IMAGES / name).as_posix(), cv2.cvtColor(ip, cv2.COLOR_RGB2BGR))
    cv2.imwrite((OUT_MASKS  / name).as_posix(), mp)

    # quick colored preview for QA
    viz = np.zeros((*mp.shape, 3), dtype=np.uint8)
    viz[mp == CLS_ROAD] = (50, 200, 255)
    viz[mp == CLS_BLDG] = (255, 255, 255)
    cv2.imwrite((OUT_VIZ / name).as_posix(), viz)

    return name, mp

# ----------------- MAIN -----------------
def main():
    # 1) Index road files by bbox (fast)
    roads_idx = index_roads_bboxes(R_DIR_SN3)
    print(f"[INFO] Indexed road files: {len(roads_idx)} from {R_DIR_SN3}")

    # 2) Collect SN2 images
    tif_files = sorted(IMG_DIR_SN2.glob("*.tif"))
    print(f"[INFO] SN2 images: {len(tif_files)} from {IMG_DIR_SN2}")

    manifest = []  # per-patch pixel counts

    for tif_path in tqdm(tif_files, desc="Processing images"):
        base_name = tif_path.stem

        # Skip if we already have patches from this tile and not overwriting
        if not OVERWRITE and any((OUT_IMAGES / f).exists() for f in (OUT_IMAGES.glob(f"{base_name}_*.png"))):
            continue

        # 3) Load image + meta
        img, meta = load_image_rgb(tif_path)
        H, W = meta["height"], meta["width"]
        out_shape = (H, W)
        transform = meta["transform"]
        img_crs   = meta["crs"]

        # 4) Rasterize buildings (SN2) by filename
        b_geo = buildings_geo_for_image(tif_path)
        m_b = rasterize_buildings(b_geo, out_shape, transform, CLS_BLDG)

        # 5) Rasterize roads (SN3) by bbox overlap
        m_r = roads_mask_for_image_bbox(tif_path, roads_idx, out_shape, transform, img_crs, ROAD_BUFFER_METERS)

        # 6) Compose single-channel mask [0=bg,1=road,2=bldg] with overlay priority
        mask = np.zeros(out_shape, dtype=np.uint8)
        for layer in CLASS_OVERLAY_ORDER:
            if layer == "road":
                mask[m_r == CLS_ROAD] = CLS_ROAD
            elif layer == "building":
                mask[m_b == CLS_BLDG] = CLS_BLDG

        # 7) Sliding-window patching with positive selection for roads
        kept = 0
        rng = np.random.default_rng(42)  # fixed for reproducibility; change/seed per run if desired

        for y in range(0, H - PATCH_SIZE + 1, STRIDE):
            for x in range(0, W - PATCH_SIZE + 1, STRIDE):
                m_patch = mask[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
                i_patch = img[y:y + PATCH_SIZE,  x:x + PATCH_SIZE]

                road_px = int(np.count_nonzero(m_patch == CLS_ROAD))
                keep = (road_px >= MIN_ROAD_PX)

                # optionally keep a small fraction of bg-only patches for stability
                if not keep and KEEP_BG_PROB > 0.0:
                    if rng.random() < KEEP_BG_PROB:
                        keep = True

                if not keep:
                    continue

                name, mp = save_patch(i_patch, m_patch, base_name, y, x)
                counts = np.bincount(mp.ravel(), minlength=3).tolist()
                manifest.append({"file": name, "bg": counts[0], "road": counts[1], "bldg": counts[2]})
                kept += 1

        if kept == 0:
            # If nothing was kept (very rare), keep at least one center crop for completeness
            cy = (H - PATCH_SIZE) // 2
            cx = (W - PATCH_SIZE) // 2
            name, mp = save_patch(img[cy:cy+PATCH_SIZE, cx:cx+PATCH_SIZE],
                                  mask[cy:cy+PATCH_SIZE, cx:cx+PATCH_SIZE],
                                  base_name, cy, cx)
            counts = np.bincount(mp.ravel(), minlength=3).tolist()
            manifest.append({"file": name, "bg": counts[0], "road": counts[1], "bldg": counts[2]})

    # 8) Save manifest with per-patch class counts
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[INFO] Saved manifest with {len(manifest)} patches at {OUT_DIR/'manifest.json'}")

    # 9) Quick summary
    if manifest:
        tot = np.array([[m["bg"], m["road"], m["bldg"]] for m in manifest]).sum(axis=0)
        ratios = tot / tot.sum()
        print(f"[SUMMARY] Patches: {len(manifest)} "
              f"| Pixels per class [bg, road, bldg]: {tot.tolist()} "
              f"| ratios: {ratios.tolist()}")

if __name__ == "__main__":
    main()
