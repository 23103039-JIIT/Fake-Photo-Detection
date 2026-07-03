# Where the labelled training images live
DATASET_DIR = "dataset"
# REAL_DIR = "real"       # images in real  -> label 0
REAL_DIR = "images"       # images in images(real)  -> label 0
SCREEN_DIR = "screen"   # images in screen -> label 1

SAVED_MODEL_DIR = "saved_model"
MODEL_FILENAME = "model.joblib"

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png")

IMAGE_SIZE = (256, 256)          # (width, height)

ENABLE_GRAYSCALE = False         # convert to gray before feature extraction
ENABLE_GAUSSIAN_BLUR = False     # mild denoise; can hurt frequency features
GAUSSIAN_BLUR_KERNEL = (3, 3)    # must be odd numbers
ENABLE_HIST_EQUALIZATION = False # boost contrast (applied on gray channel)
ENABLE_NORMALIZATION = True      # scale pixel values to 0..1 range

ENABLE_FFT = True        # frequency-domain features (screen moiré / grids)
ENABLE_LBP = True        # Local Binary Pattern texture features
ENABLE_GLCM = True       # Gray-Level Co-occurrence Matrix texture features
ENABLE_EDGE = True       # Canny / Sobel / Laplacian edge features
ENABLE_COLOR = True      # RGB / HSV colour statistics
ENABLE_REFLECTION = True # glare / specular highlight features

# --- LBP ---
LBP_RADIUS = 1                       # neighbourhood radius in pixels
LBP_N_POINTS = 8 * LBP_RADIUS        # number of sampling points on the circle
LBP_METHOD = "uniform"               # rotation-invariant, compact histogram

# --- GLCM ---
GLCM_DISTANCES = [1, 2]              # pixel-pair distances
GLCM_ANGLES = [0, 45, 90, 135]       # directions in degrees (converted later)
GLCM_LEVELS = 32                     # gray levels (down-quantised for speed)

# --- FFT ---
FFT_LOW_FREQ_RADIUS_RATIO = 0.15
FFT_NUM_PEAKS = 5                    # how many dominant frequency peaks to keep

# --- Edge ---
CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150

# --- Reflection / glare ---
REFLECTION_BRIGHTNESS_THRESHOLD = 240

CLASSIFIER = "RandomForest"
# CLASSIFIER = "LogisticRegression"
# CLASSIFIER = "SVM"

# Random Forest hyper-parameters
RF_N_ESTIMATORS = 300
RF_MAX_DEPTH = None
RF_RANDOM_STATE = 42

# Logistic Regression hyper-parameters
LR_MAX_ITER = 1000
LR_C = 1.0

# SVM hyper-parameters
SVM_C = 1.0
SVM_KERNEL = "rbf"
SVM_GAMMA = "scale"

TEST_SPLIT_RATIO = 0.2 
RANDOM_SEED = 42 
