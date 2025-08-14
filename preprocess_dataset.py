import json
from tqdm import tqdm
from preprocess_utils import *
from config import *
import numpy as np
from config import OUT_DIR, OVERWRITE

def main(limit_images=None, overwrite=None):
    if overwrite is None:
        overwrite = OVERWRITE  # z config.py

    # 1) Index road files by bbox (fast)
    roads_idx = index_roads_bboxes(R_DIR_SN3)
    print(f"[INFO] Indexed road files: {len(roads_idx)} from {R_DIR_SN3}")

    # 2) Collect SN2 images
    tif_files = sorted(IMG_DIR_SN2.glob("*.tif"))
    if limit_images is not None:
        tif_files = tif_files[:limit_images]
    print(f"[INFO] SN2 images: {len(tif_files)} from {IMG_DIR_SN2}")

    manifest = []  # per-patch pixel counts

    for tif_path in tqdm(tif_files, desc="Processing images"):
        base_name = tif_path.stem

        # Skip if we already have patches from this tile and not overwriting
        if not overwrite and any(OUT_IMAGES.glob(f"{base_name}_*.png")):
            continue

        # 3) Load image + meta
        img, meta = load_image_rgb(tif_path)
        H, W = meta["height"], meta["width"]
        out_shape = (H, W)
        transform = meta["transform"]
        img_crs   = meta["crs"]

        # 4) Rasterize buildings (SN2) by filename
        b_geo = buildings_geo_for_image(tif_path, B_DIR_SN2)
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

                name, mp = save_patch(i_patch, m_patch, base_name, y, x, OUT_IMAGES, OUT_MASKS, OUT_VIZ,
                                      patch_size=PATCH_SIZE, resize_to=RESIZE_PATCH_TO)
                counts = np.bincount(mp.ravel(), minlength=3).tolist()
                manifest.append({"file": name, "bg": counts[0], "road": counts[1], "bldg": counts[2]})
                kept += 1

        if kept == 0:
            # If nothing was kept (very rare), keep at least one center crop for completeness
            cy = (H - PATCH_SIZE) // 2
            cx = (W - PATCH_SIZE) // 2
            name, mp = save_patch(
                img[cy:cy + PATCH_SIZE, cx:cx + PATCH_SIZE],
                mask[cy:cy + PATCH_SIZE, cx:cx + PATCH_SIZE],
                base_name, cy, cx,
                OUT_IMAGES, OUT_MASKS, OUT_VIZ,
                patch_size=PATCH_SIZE, resize_to=RESIZE_PATCH_TO)

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
    import argparse
    from config import OVERWRITE as CFG_OVERWRITE

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images to process")
    parser.add_argument("--overwrite", action="store_true", help="Force overwrite existing patches")
    args = parser.parse_args()

    main(limit_images=args.limit, overwrite=(args.overwrite or CFG_OVERWRITE))
