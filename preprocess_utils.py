from pathlib import Path
import re
import warnings
import cv2
import numpy as np
import geopandas as gpd
import fiona
from shapely.geometry import box
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_bounds

CLS_BG, CLS_ROAD, CLS_BLDG = 0, 1, 2

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
    """Read RGB image and meta info from a GeoTIFF."""
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

def buildings_geo_for_image(tif_path: Path, buildings_dir: Path) -> Path | None:
    """Get path to building geojson matching a given image."""
    stem = tif_path.stem
    tif_id = re.sub(r"^RGB[-_]?PanSharpen[_-]?", "", stem, flags=re.I)
    cand = buildings_dir / f"buildings_{tif_id}.geojson"
    return cand if cand.exists() else None

def rasterize_buildings(geo_path: Path, out_shape, transform, class_value=CLS_BLDG):
    """Rasterize building polygons."""
    if geo_path is None or not geo_path.exists():
        return np.zeros(out_shape, dtype=np.uint8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gdf = gpd.read_file(geo_path)
    if gdf.empty:
        return np.zeros(out_shape, dtype=np.uint8)
    shapes = [(geom, class_value) for geom in gdf.geometry if geom and not geom.is_empty]
    m = rasterize(shapes=shapes, out_shape=out_shape, transform=transform,
                  fill=0, all_touched=True, dtype=np.uint8)
    m[m > 0] = class_value
    return m

def index_roads_bboxes(roads_root: Path) -> gpd.GeoDataFrame:
    """Index road geojsons by bounding box."""
    paths = list(roads_root.rglob("*.geojson"))
    recs = []
    for p in paths:
        try:
            with fiona.open(p.as_posix()) as src:
                b = src.bounds
                crs = src.crs or {"init": "epsg:4326"}
            recs.append({"path": str(p), "geometry": box(*b), "src_crs": crs})
        except Exception as e:
            print(f"[WARN] Could not read bounds for {p.name}: {e}")
    if not recs:
        return gpd.GeoDataFrame(columns=["path", "geometry", "src_crs"], geometry="geometry", crs="EPSG:4326")
    gdf = gpd.GeoDataFrame(recs, geometry="geometry", crs="EPSG:4326")
    gdf = gdf.to_crs(4326)
    _ = gdf.sindex
    return gdf

def roads_mask_for_image_bbox(img_path: Path, roads_index_gdf: gpd.GeoDataFrame,
                              out_shape, transform, img_crs,
                              road_buffer_m=5.0) -> np.ndarray:
    """Find and rasterize roads overlapping the image bbox."""
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
            g = g.set_crs(4326, allow_override=True).to_crs(img_crs)
            parts.append(g)
        except Exception as e:
            print(f"[WARN] Read failed for {p.name}: {e}")
    if not parts:
        return np.zeros(out_shape, dtype=np.uint8)

    roads = gpd.pd.concat(parts, ignore_index=True)
    if roads.empty:
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
    m = rasterize(shapes=shapes, out_shape=out_shape, transform=transform,
                  fill=0, all_touched=True, dtype=np.uint8)
    m[m > 0] = CLS_ROAD
    return m

def save_patch(img_patch: np.ndarray, mask_patch: np.ndarray, base_name: str, y: int, x: int,
               out_images: Path, out_masks: Path, out_viz: Path,
               cls_bg=CLS_BG, cls_road=CLS_ROAD, cls_bldg=CLS_BLDG,
               patch_size=None, resize_to=None):
    """
    Save image/mask patch and optional colored preview.
    Returns: (filename, mask_patch)
    """
    name = f"{base_name}_{y}_{x}.png"
    ip = img_patch
    mp = mask_patch

    # Optional resize
    if resize_to is not None and resize_to != patch_size:
        ip = cv2.resize(ip, (resize_to, resize_to), interpolation=cv2.INTER_AREA)
        mp = cv2.resize(mp, (resize_to, resize_to), interpolation=cv2.INTER_NEAREST)

    # Save image & mask
    cv2.imwrite((out_images / name).as_posix(), cv2.cvtColor(ip, cv2.COLOR_RGB2BGR))
    cv2.imwrite((out_masks / name).as_posix(), mp)

    # Colored preview for QA
    viz = np.zeros((*mp.shape, 3), dtype=np.uint8)
    viz[mp == cls_road] = (50, 200, 255)   # light blue for roads
    viz[mp == cls_bldg] = (255, 255, 255)  # white for buildings
    cv2.imwrite((out_viz / name).as_posix(), viz)

    return name, mp
