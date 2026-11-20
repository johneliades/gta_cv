# GTA V Computer Vision Aimbot

Real-time object detection for GTA V using **YOLOv4-tiny** with OpenCV's DNN module. Detects pedestrians and provides aim assistance with configurable smoothing.

<p align="center">
  <img src="docs/preview.gif" alt="animated" />
</p>

## Features

- **Real-time detection** — YOLOv4-tiny running via OpenCV DNN (CUDA or CPU)
- **Smooth aiming** — Configurable interpolation for natural mouse movement
- **On-screen HUD** — Live FPS counter, detection count, crosshair, and aim status
- **Auto-fire** — Automatically fires when crosshair is over a target
- **Configurable** — All settings in `config.py` (thresholds, keys, model, etc.)
- **Target priority** — Automatically selects the closest person to crosshair

## Requirements

- Python 3.7+
- Windows OS
- GTA V (running in **windowed mode**)
- [YOLOv4-tiny weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights) — place in `models/`
- (Optional) NVIDIA GPU with CUDA support for acceleration

## Installation

```bash
git clone https://github.com/johneliades/gta_cv.git
cd gta_cv
pip install -r requirements.txt        # CPU (pip OpenCV)
# or, with your own CUDA-enabled OpenCV build already installed:
pip install -r requirements-gpu.txt    # does not install/overwrite OpenCV
```

Download the YOLOv4-tiny weights file into `models/`:
```bash
curl -L -o models/yolov4-tiny.weights https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights
```

## Configuration

Edit `config.py` to customize behavior:

| Setting | Default | Description |
|---------|---------|-------------|
| `USE_CUDA` | `True` | Enable GPU acceleration (set `False` for CPU) |
| `BLOB_SIZE` | `(416, 416)` | Input resolution (higher = more accurate, slower) |
| `CONFIDENCE_THRESHOLD` | `0.5` | Minimum detection confidence |
| `NMS_THRESHOLD` | `0.3` | Non-maxima suppression overlap threshold |
| `AIM_SMOOTHING` | `0.6` | Mouse smoothing (0.0–1.0, lower = smoother) |
| `AIM_ACTIVATION_KEY` | `X` | Key to hold for aim assist |
| `AUTO_FIRE` | `True` | Auto-click when on target |
| `TARGET_CLASS` | `person` | COCO class to target |

## Usage

1. Launch GTA V in **windowed mode**
2. Run the script:
   ```bash
   python main.py
   ```
3. Hold **X** to activate aim assist
4. Press **Q** on the preview window to quit

## Controls

| Key | Action |
|-----|--------|
| Hold `X` | Activate aimbot |
| `Q` (on preview window) | Quit |

## Project Structure

```
├── main.py               # Entry point: main loop (capture → detect → aim → draw)
├── config.py             # All configurable settings
├── aimbot/
│   ├── capture.py        # Game window lookup + screen capture
│   ├── detector.py       # YOLO model loading and detection (OpenCV DNN)
│   ├── targeting.py      # Self-exclusion and target selection
│   ├── controls.py       # Key polling, mouse movement, fire button
│   ├── hud.py            # Preview overlay drawing
│   └── keys.py           # DirectInput keyboard/mouse driver
├── models/
│   ├── coco.names        # COCO dataset class labels (80 classes)
│   ├── yolov4-tiny.cfg   # YOLOv4-tiny architecture (+ .weights, downloaded)
│   └── yolov4.cfg        # YOLOv4 full architecture (optional, + .weights)
├── docs/preview.gif
├── requirements.txt      # Dependencies incl. CPU OpenCV
├── requirements-gpu.txt  # Dependencies without OpenCV (custom CUDA build)
└── README.md
```

## Author

**Eliades John** — [GitHub](https://github.com/johneliades)
