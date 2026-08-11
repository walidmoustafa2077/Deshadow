"""Configuration for LP-IOANet (from the LP-IOANet paper + Mixed_Shadow_Dataset).

The Mixed_Shadow_Dataset is portrait 4:3:
  - Mixed_Shadow_Dataset_256x192  -> low-res core (H=256, W=192)
  - Mixed_Shadow_Dataset_1024x768 -> high-res target (H=1024, W=768)
"""

# Resolutions (H, W) — portrait 4:3 to match the dataset
LOW_RES = (256, 192)      # IOANet core resolution (256x192)
HIGH_RES = (1024, 768)    # target high-resolution output (1024x768)
INTERMEDIATE_RES = (512, 384)  # residual refinement resolution

# Model
PYRAMID_LEVELS = 2
UPSAMPLER_HIDDEN = 16
COORD_ATTN_REDUCTION = 32
PRETRAINED_ENCODER = True

# Dataset paths (Kaggle)
DATA_ROOT = "Mixed_Shadow_Dataset_1024x768"  # high-res version for stage 2
LOW_RES_DATA_ROOT = "Mixed_Shadow_Dataset_256x192"  # low-res version for stage 1
TRAIN_CSV = "train_metadata.csv"
TEST_CSV = "test_metadata.csv"

# Stage 1 (IOANet training)
STAGE1_EPOCHS = 1000
STAGE1_LR = 2e-4
L1_WEIGHT = 10.0
LPIPS_WEIGHT = 5.0
BATCH_MIX = {"A-BSDD": 15, "Doc3DS+": 15, "A-OSR": 2}  # not used with this dataset

# Stage 2 (upsampler training)
STAGE2_EPOCHS = 200
STAGE2_LR = 2e-4
STAGE2_LOSS = "l1"

# Optimizer
OPTIMIZER = "adam"
BATCH_SIZE = 8
NUM_WORKERS = 4

# Training / monitoring (matching reference DocShadow-Lite)
VAL_INTERVAL = 5          # validate every N epochs
CKPT_INTERVAL = 50        # save periodic checkpoint every N epochs
EARLY_STOP_PATIENCE = 50  # Stage 1 patience
EARLY_STOP_PATIENCE_S2 = 30  # Stage 2 patience
GRAD_CLIP = 1.0           # gradient clipping max norm
# Scheduler: ReduceLROnPlateau (epoch-count agnostic, seamless resume)
LR_SCHEDULER = "reduce_on_plateau"  # Stage 1
LR_SCHEDULER_S2 = "reduce_on_plateau"  # Stage 2
PLATEAU_FACTOR = 0.5      # halve LR on plateau
PLATEAU_PATIENCE = 10     # validation checks before reducing
PLATEAU_COOLDOWN = 5      # epochs to wait after a reduction
PLATEAU_MIN_LR = 1e-6     # LR floor
DEBUG = True              # save debug sample images

# Output directories
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"
SAMPLE_DIR = "samples"
