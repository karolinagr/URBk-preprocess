# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is a SpaceNet satellite data preprocessing pipeline that converts raw Paris satellite imagery and vector annotations into ML-ready datasets for urban analysis. The pipeline processes SpaceNet SN2 (buildings) and SN3 (roads) datasets to create paired satellite images and segmentation masks.

## Key Commands

### Running the Preprocessing Pipeline
```bash
# Process all images with default settings
python preprocess_dataset.py

# Process limited number of images (useful for testing)
python preprocess_dataset.py --limit 10

# Force overwrite existing patches
python preprocess_dataset.py --overwrite

# Combine flags
python preprocess_dataset.py --limit 50 --overwrite
```

### Data Validation
```bash
# Check mask class distribution
python check_masks.py
```

### Environment Setup
```bash
# Create virtual environment
python -m venv .venv

# Windows activation
.venv\Scripts\activate

# Git Bash activation
source .venv/Scripts/activate

# Install minimal dependencies
pip install -r requirements.txt

# Install full development environment
pip install -r requirements_full.txt
```

## Architecture Overview

### Data Flow Pipeline
1. **Input Data Sources**:
   - `SN2_buildings_train_AOI_3_Paris/`: Satellite images (.tif) and building vectors (.geojson)
   - `SN3_roads_train_AOI_3_Paris/`: Road vectors (.geojson)

2. **Core Processing** (`preprocess_dataset.py`):
   - Loads satellite images and extracts metadata
   - Rasterizes building footprints by filename matching
   - Rasterizes roads by bounding box spatial overlap
   - Combines into single 3-class mask: background(0), roads(1), buildings(2)
   - Extracts 256×256 patches with 128px stride
   - Filters patches to prioritize road-containing samples

3. **Output Structure**:
   - `dataset_paris_patches/images/`: RGB satellite patches
   - `dataset_paris_patches/masks/`: Corresponding segmentation masks
   - `dataset_paris_patches/viz/`: Optional visualization overlays
   - `dataset_paris_patches/manifest.json`: Per-patch class statistics

### Key Configuration Parameters (`config.py`)
- **PATCH_SIZE**: 256px patches
- **STRIDE**: 128px overlap between patches
- **MIN_ROAD_PX**: 200 minimum road pixels to keep a patch
- **ROAD_BUFFER_METERS**: 5m buffer for road rasterization
- **CLASS_OVERLAY_ORDER**: Buildings overlay roads in final masks

### Utility Functions (`preprocess_utils.py`)
- **Image Processing**: Load GeoTIFF, stretch to uint8, patch extraction
- **Geospatial Operations**: Rasterize vectors, coordinate transformations, bounding box operations
- **Road Indexing**: Spatial indexing for efficient road-image matching across datasets

## Important Implementation Details

### Spatial Coordinate Handling
The pipeline handles coordinate system differences between SN2 (images/buildings) and SN3 (roads) datasets by transforming all geometries to the image CRS before rasterization.

### Memory Management
Images are processed sequentially to manage memory usage. The pipeline supports resuming interrupted runs by checking for existing patches (unless `--overwrite` is used).

### Class Overlay Strategy
When roads and buildings overlap in the same pixel, buildings take priority based on `CLASS_OVERLAY_ORDER` configuration. This ensures consistent mask generation.

### Patch Selection Strategy
The pipeline uses positive sampling to ensure road coverage:
- Keeps patches with ≥ MIN_ROAD_PX road pixels
- Randomly keeps 10% of background-only patches for training stability
- Guarantees at least one patch per image (center crop fallback)