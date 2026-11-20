# Configuration file for GTA V CV Aimbot
# Adjust these values to tune detection and aiming behavior.

import os

# Model files live in models/ next to this file (works from any working dir)
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# ─── Window Settings ───────────────────────────────────────────────────────────
WINDOW_TITLE = "Grand Theft Auto V"
INITIAL_WINDOW_X = 0
INITIAL_WINDOW_Y = 0
INITIAL_WINDOW_WIDTH = 900
INITIAL_WINDOW_HEIGHT = 800

# ─── YOLO Model Settings ──────────────────────────────────────────────────────
# For full YOLOv4 (more accurate, much slower) use yolov4.cfg / yolov4.weights
MODEL_CFG = os.path.join(MODELS_DIR, "yolov4-tiny.cfg")
MODEL_WEIGHTS = os.path.join(MODELS_DIR, "yolov4-tiny.weights")
CLASSES_FILE = os.path.join(MODELS_DIR, "coco.names")

# Use CUDA acceleration (set to False to use CPU)
USE_CUDA = True

# Input blob size (must match model training size; multiples of 32)
BLOB_SIZE = (416, 416)

# ─── Detection Settings ───────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.3

# Target class name to aim at
TARGET_CLASS = "person"

# ─── Aiming Settings ─────────────────────────────────────────────────────────
# Key that activates the aimbot (hold to aim)
AIM_ACTIVATION_KEY = "X"

# Smoothing factor for mouse movement (0.0 = no move, 1.0 = instant snap)
# Lower values give smoother but slower tracking
AIM_SMOOTHING = 0.6

# Auto-fire when crosshair is on target
AUTO_FIRE = True

# ─── Self-Exclusion ──────────────────────────────────────────────────────────
# Your own character is also detected as a "person". These rules ignore it.
# All values are fractions of the game view (0.0 = left/top, 1.0 = right/bottom).
SELF_EXCLUSION_ENABLED = True

# Only ONE box per frame is treated as self: the largest one that matches the
# third-person OR the first-person rule. Position is judged by the box's
# bottom-centre point (middle of its bottom edge).

# Third person: your character, bottom-left of the crosshair.
# Drawn as a yellow rectangle in the preview.
SELF_THIRD_PERSON_MIN_HEIGHT = 0.25   # box height >= this fraction of screen
SELF_THIRD_PERSON_MIN_BOTTOM = 0.8    # bottom-centre in the bottom 20%...
SELF_THIRD_PERSON_MAX_X = 0.55        # ...and in the left 55%

# First person: arms + gun, bottom middle (a smaller box when aiming).
# Drawn as a magenta rectangle in the preview.
SELF_FIRST_PERSON_MIN_HEIGHT = 0.08   # gun/hands box can be short
SELF_FIRST_PERSON_MIN_BOTTOM = 0.85   # bottom-centre in the bottom 15%...
SELF_FIRST_PERSON_HALF_WIDTH = 0.25   # ...and within 25% left/right of centre

# Debug: print each person box's height and bottom-centre position (as
# fractions) next to it in the preview, to help tune the values above.
SHOW_SELF_DEBUG = True

# ─── Display Settings ─────────────────────────────────────────────────────────
SHOW_FPS = True
SHOW_DETECTION_COUNT = True
WINDOW_NAME = "GTA CV Detection"

