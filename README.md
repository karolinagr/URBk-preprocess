# SpaceNet Data Preprocessing for Urban Analysis
This repository contains scripts for preparing the SpaceNet dataset for semantic segmentation tasks, 
such as building footprint detection in Paris. The output is a paired dataset of satellite images and binary masks, 
ready for training segmentation models (e.g., U-Net).

## Project Scope
This preprocessing stage corresponds to **Step 1** of the full URBk_AI project:  
> Converting raw SpaceNet satellite imagery and building footprint data into a machine-learning-friendly format.

The output dataset will be used for:
- Urban spatial analysis (building density, green space ratio, vacant plots)
- Training deep learning models for semantic segmentation

## Folder Structure
```
URBk_AI_preprocess/
├── preprocess_dataset.py    # Main preprocessing script
├── preprocess_utils.py 
├── config.py
├── check_masks.py  
├── .gitignore
├── README.md
├── requirements.txt
├── requirements_full.txt  
│
├── dataset_paris_patches/    # Output dataset (sample, images + masks) [IGNORED by Git]
│   ├── images/
│   ├── masks/
│   ├── viz/
│   └── manifest.json
│
└── SN2_buildings_train_AOI_3_Paris/    # Raw SpaceNet data [IGNORED by Git]
│   └── AOI_3_Paris_Train
│       ├── geojson/buildings/ 
│       ├── RGB-PanSharpen/
│       └── summaryData/AOI_3_Paris_Building_Solution.csv
│
└──SN3_roads_train_AOI_3_Paris/    # Raw SpaceNet data [IGNORED by Git]
    └── AOI_3_Paris
        ├── geojson_roads/ 
        └── PS-RGB/
```
## Requirements
Minimal required packages:\
numpy\
pandas\
tqdm\
shapely\
rasterio\
opencv-python

Use `requirements.txt` for a clean install of just the essentials.
Use `requirements_full.txt` if you need an exact replica of my development environment.

## How to Use
1. Download SpaceNet Data
- Go to: https://spacenet.ai/sn2-buildings-dataset/
- Download the `SN2_buildings_train_AOI_3_Paris package` and `SN3_roads_train_AOI_3_Paris`.
- Extract it into the root of this preprocessing project:
`preprocessing/SN2_buildings_train_AOI_3_Paris/`
`preprocessing/SN3_roads_train_AOI_3_Paris/`

The folders should contain:
```
SN2_buildings_train_AOI_3_Paris/
├── RGB-PanSharpen/
└── geojson/buildings/
```

```
SN3_roads_train_AOI_3_Paris/    
└── AOI_3_Paris
   ├── geojson_roads/ 
   └── PS-RGB/
```

2. (Optional) Create a Virtual Environment
If using PyCharm, .venv is created automatically.
Manually:
`python -m venv .venv`\
`source .venv/Scripts/activate`    # Windows PowerShell / Git Bash

3. Run preprocessing.
`python preprocess_spacenet_paris.py`

The script will:
- Read .tif satellite imagery
- Load matching .geojson building footprints
- Generate binary masks from vector data
- Save matching images/ and masks/ pairs to dataset_paris_small/

4. Output Example
```
dataset_paris_patches/
├── images/
│   ├── RGB-PanSharpen_AOI_3_Paris_img3_128_0.png
│   ├── ...
│
└── masks/
│   ├── RGB-PanSharpen_AOI_3_Paris_img3_128_0.png
│   ├── ...
│
└── viz/
│   ├── RGB-PanSharpen_AOI_3_Paris_img3_128_0.png
│   ├── ...
│
└── manifest.json
```
Each image in `images/` has a corresponding binary mask in `masks/`.

## Notes
- For testing purposes, the current script is limited to 300 samples.
- To process the full dataset, adjust the LIMIT variable in the script.
- Large datasets may require more memory and disk space.

## Sources & Inspiration
- SpaceNet 2 Building Detection Dataset — https://spacenet.ai/sn2-buildings-dataset/
- Rasterio and Shapely documentation

## Autor
Project developed as part of an AI/ML portfolio (2025).
LinkedIn: https://linkedin.com/in/karolina-groszek 
GitHub: https://github.com/-github

