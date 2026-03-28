import os
import torch

# ================== Path Configuration ==================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.join(ROOT, "dataset", "processed")
RAW_DIR = os.path.join(ROOT, "dataset", "raw")
IMAGE_DIR = os.path.join(DATA_DIR, "images_all")

MODEL_DIR = os.path.join(ROOT, "models")
RESULTS_DIR = os.path.join(ROOT, "results")

TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
VAL_CSV = os.path.join(DATA_DIR, "val.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")

MODEL_PATH = os.path.join(MODEL_DIR, "resnet18_chestxray.pth")

# ================== Model Configuration ==================
IMG_SIZE = 224
NUM_CLASSES = 2
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ================== Training Configuration ==================
BATCH_SIZE = 32
NUM_WORKERS = 4
EPOCHS = 10
LR = 1e-4
WEIGHT_DECAY = 1e-4

# ================== Attack Configuration ==================
EPS_LIST = [0.0, 0.25/255, 0.5/255, 0.75/255, 1/255, 2/255]
PGD_STEPS = 10
PGD_ALPHA = 1/255

CW_C = 1.0
CW_KAPPA = 0
CW_STEPS = 100
CW_LR = 0.01

# ================== Device Configuration ==================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dirs():
    """Create necessary directories if they don't exist."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)


def get_device():
    """Return the current device."""
    return DEVICE


def print_config():
    """Print current configuration."""
    print("=" * 50)
    print("Configuration")
    print("=" * 50)
    print(f"Device: {DEVICE}")
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"\nPaths:")
    print(f"  ROOT: {ROOT}")
    print(f"  DATA_DIR: {DATA_DIR}")
    print(f"  MODEL_DIR: {MODEL_DIR}")
    print(f"  RESULTS_DIR: {RESULTS_DIR}")
    print(f"\nTraining:")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning rate: {LR}")
    print(f"\nAttack:")
    print(f"  Epsilon list: {EPS_LIST}")
    print(f"  PGD steps: {PGD_STEPS}")
    print("=" * 50)
