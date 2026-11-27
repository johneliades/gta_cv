# Configuration file for GTA V CV Aimbot
# Adjust these values to tune detection and aiming behavior.

import os

# Model files live in models/ next to this file (works from any working dir)
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# ─── Capture Settings ──────────────────────────────────────────────────────────
# The target is always the active (foreground) window, followed as you switch.
# The terminal/IDE the script was started from is never targeted; while it
# (or the desktop) is active, the overlay hides and aiming pauses.

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

# Aim point: the target's head.
# If the model has this class (a custom-trained model), a detected head inside
# the upper half of a person box is used. COCO models have none, so the head
# is estimated from the person box instead:
HEAD_CLASS = "head"
# ...centred horizontally, this fraction of the box height below its top
# (head centre ~1/16 of body height down; a bit more for YOLO's loose boxes).
HEAD_Y_FRACTION = 0.08
# ...as a square this fraction of the box height (a head is ~1/8 of body
# height; slightly larger to forgive box jitter). Auto-fire only triggers
# while the crosshair is inside this square (or inside a detected head box).
# Drawn as a small red square on the current target.
HEAD_SIZE_FRACTION = 0.14

# ─── Aiming Settings ─────────────────────────────────────────────────────────
# Key that activates the aimbot (hold to aim)
AIM_ACTIVATION_KEY = "X"

# Smoothing factor for mouse movement (0.0 = no move, 1.0 = instant snap)
# Lower values give smoother but slower tracking
AIM_SMOOTHING = 0.6

# Auto-fire while the crosshair is on the target's head area (see HEAD_SIZE_FRACTION)
AUTO_FIRE = True

# Target lock: while the aim key is held, the person closest to the crosshair
# when aiming started stays the target. Detection can miss a person for a few
# frames; the lock waits (no aiming or firing) up to this many frames in a row
# for them to reappear before moving on to the next closest person.
TRACK_MAX_MISSES = 12

# ─── Self-Exclusion ──────────────────────────────────────────────────────────
# Your own character is also detected as a "person" and must never be aimed at.
# Each person box gets a self score from three signals (see aimbot/self_id.py):
#   motion      stays fixed on screen while the camera turns (background moves)
#   appearance  colours match the saved profile of your character
#   position    lies in the self areas below (a weak hint)
# Only the highest-scoring box counts as self, and only at or above
# SELF_SCORE_THRESHOLD. A signal with no evidence yet counts as 0.5.
SELF_EXCLUSION_ENABLED = True

SELF_WEIGHT_MOTION = 0.45
SELF_WEIGHT_APPEARANCE = 0.40
SELF_WEIGHT_POSITION = 0.15
# With nothing else known: 0.425 + 0.15 if in a self area = 0.575, so the
# position rule alone still decides (the fallback) until there is more evidence.
SELF_SCORE_THRESHOLD = 0.5
SELF_STICKY_BONUS = 0.05   # added to last frame's self box, against flicker

# Pick yourself by hand (keys: a letter/digit, "F1".."F12", or "ENTER",
# "SPACE", "TAB", ...; see controls.py): press the next key to start, press it
# again to move the highlight through the people on screen (left to right),
# then a Cancel placeholder, then round again. The confirm key picks the
# highlighted person (or leaves without a pick on Cancel).
# The pick is self for as long as it stays tracked, and its colours become the
# saved profile. Aiming pauses while selecting.
SELF_SELECT_NEXT_KEY = "F8"
SELF_SELECT_CONFIRM_KEY = "ENTER"

# Motion signal: background shift by phase correlation on a small greyscale
# copy of the frame (this many pixels wide).
SELF_MOTION_WIDTH = 160
SELF_MOTION_MIN_SHIFT = 0.006      # camera counts as turning above this fraction of frame width per frame
SELF_MOTION_MIN_RESPONSE = 0.1     # below this the shift is unreliable and ignored
SELF_MOTION_SMOOTHING = 0.3        # running-average rate of each box's stillness
SELF_MOTION_MIN_SAMPLES = 5        # turning frames before a box's motion score is fully trusted
SELF_TRACK_MIN_IOU = 0.3           # overlap needed to follow a box to the next frame

# Appearance signal: HSV colour histogram similarity (0..1) to the profile,
# mapped so NO_MATCH and below scores 0 and MATCH and above scores 1.
SELF_APPEARANCE_NO_MATCH = 0.35
SELF_APPEARANCE_MATCH = 0.75
# The profile is kept between runs in this file (None: never saved)
SELF_PROFILE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "self_profile.npy")
# Learn/refresh the profile on its own from the self box while the camera
# turns and the motion signal alone is at least this sure (follows outfit and
# lighting changes; the first profile can be learned this way too).
SELF_APPEARANCE_AUTO_UPDATE = True
SELF_AUTO_UPDATE_MIN_MOTION = 0.8
SELF_AUTO_UPDATE_RATE = 0.05

# Position signal: the self areas. All values are fractions of the game view
# (0.0 = left/top, 1.0 = right/bottom), judged by the box's bottom-centre
# point (middle of its bottom edge).

# Third person: your character, bottom-left of the crosshair.
SELF_THIRD_PERSON_MIN_HEIGHT = 0.25   # box height >= this fraction of screen
SELF_THIRD_PERSON_MIN_BOTTOM = 0.8    # bottom-centre in the bottom 20%...
SELF_THIRD_PERSON_MAX_X = 0.55        # ...and in the left 55%

# First person: arms + gun, bottom middle (a smaller box when aiming).
SELF_FIRST_PERSON_MIN_HEIGHT = 0.08   # gun/hands box can be short
SELF_FIRST_PERSON_MIN_BOTTOM = 0.85   # bottom-centre in the bottom 15%...
SELF_FIRST_PERSON_HALF_WIDTH = 0.25   # ...and within 25% left/right of centre

# Debug: print each person box's height and bottom-centre position (as
# fractions), and its self score (m=motion a=appearance p=position = total),
# next to it in the overlay, to help tune the values above.
SHOW_SELF_DEBUG = True

# ─── Display Settings ─────────────────────────────────────────────────────────
SHOW_FPS = True
SHOW_DETECTION_COUNT = True

# Restart as administrator at launch (Windows asks for permission). Needed for
# games that run as administrator, e.g. GTA IV: Windows hides their key
# presses from normal programs, so no key (aim, quit, selection) would work.
# The administrator copy runs in its own window.
RUN_AS_ADMIN = True

# The HUD is drawn directly over the game (a click-through overlay window),
# so there is no separate window to press a key in. Hold this key to quit
# (a letter/digit like "Q", or a function key like "F10"). Ctrl+C also works.
QUIT_KEY = "F10"

