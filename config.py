from pathlib import Path

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
