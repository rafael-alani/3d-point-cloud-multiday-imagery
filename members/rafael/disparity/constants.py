import os
import glob

# First 10 pairs based on the heuristic criterion
N = 10

# Remove for faster compute
IS_DEBUG_MODE = True
IS_DEBUG_PAIR = False
IS_ONE_RANDOM_PAIR = True

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '../../../data')
WV3_PATH = os.path.join(DATA_DIR, 'rafael/WV3/PAN')

# Find StereoPipeline installation dynamically
_asp_dirs = glob.glob(os.path.join(BASE_DIR, '../../../external/StereoPipeline-*/bin'))
ASP_BIN_PATH = _asp_dirs[0] if _asp_dirs else None

#San Fernando, Argentina - target site of WV3 dataset
TARGET_LAT = -34.490278
TARGET_LON = -58.584444
H_RANGE = (0, 50) 

# Processing Parameters
TILE_SIZE = 1000

PAIR_DECENT_RESULTS = [
    ['05JAN15WV031000015JAN05135727-P1BS-500497282040_01_P001_________AAE_0AAAAABPABR0.NTF',
    '06FEB15WV031000015FEB06141035-P1BS-500497283080_01_P001_________AAE_0AAAAABPABP0.NTF'
    ]
]

# CACHES / TEMP PATHS
TEMP_PATH = os.path.join(DATA_DIR, 'TEMP')
os.makedirs(TEMP_PATH, exist_ok=True)
CACHE_PRE_PROCESSING = False
CACHE_PRE_PROCESSING_DIR = os.path.join(TEMP_PATH, 'pre_processed_images')
os.makedirs(CACHE_PRE_PROCESSING_DIR, exist_ok=True)
CACHE_DISPARITY = False
CACHE_DISPARITY_DIR = os.path.join(TEMP_PATH, 'disparity')
os.makedirs(CACHE_DISPARITY_DIR, exist_ok=True)
TMP_CROPPED_IMAGES_PATH = os.path.join(TEMP_PATH, 'cropped_images')
os.makedirs(TMP_CROPPED_IMAGES_PATH, exist_ok=True)
TMP_STEREO_OUTPUT_PATH = os.path.join(TEMP_PATH, 'stereo_output')
os.makedirs(TMP_STEREO_OUTPUT_PATH, exist_ok=True)
TMP_DISPARITY_DEBUG_PATH = os.path.join(TEMP_PATH, 'disp_debug')
os.makedirs(TMP_DISPARITY_DEBUG_PATH, exist_ok=True)

# due to opencv, this alos has to be a multiple of 16
#https://stackoverflow.com/questions/65357601/how-a-good-depth-map-can-be-created-with-stereo-cameras

MAX_DISP = 288
if MAX_DISP % 16 != 0:
    MAX_DISP += 16 - (MAX_DISP % 16)
MAX_DISP = int(MAX_DISP)
BLOCK_SIZE_DISP = 15  # block size for SGBM
WLS_LAMBDA = 8000.0 # default for now
# 0.25 very jittery
# 0.35 all right
# WLS_SIGMA = 0.45
WLS_SIGMA = 1.0
MARGIN_UNDEFINED = 24
