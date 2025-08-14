import cv2
import numpy as np
from pathlib import Path
from collections import Counter

MASKS_DIR = Path("dataset_paris_patches/masks")

counts = Counter()

for mask_path in MASKS_DIR.glob("*.png"):
    m = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    uniq = tuple(np.unique(m))
    counts[uniq] += 1

for classes, num_files in counts.items():
    print(f"Classes {classes}: {num_files} files")
