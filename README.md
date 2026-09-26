# Hand Gesture Detection using ML

An experimental computer-vision project for recognizing hand gestures with MediaPipe and OpenCV. It includes two related workflows:

- A **landmark dataset and classifier pipeline** that extracts 21 hand landmarks, augments the dataset, and trains a Random Forest model.
- A **real-time gesture-command pipeline** that detects hand landmarks, optionally uses YOLO to locate a hand region, and maps finger states to control-style commands.

> This repository is intended for local experimentation. The command labels are simulated by default; connecting them to a real drone or other device requires adding and testing the appropriate SDK integration.

## Features

- MediaPipe Tasks hand-landmark detection
- OpenCV image, video, and webcam processing
- Optional YOLO hand-region detection with `yolov8n.pt`
- Batch processing for folders of images
- Landmark CSV generation and simple Gaussian-noise data augmentation
- Random Forest gesture classifier training and persistence with Joblib
- Real-time command labels for common hand poses

## Requirements

- Python 3.12 recommended
- A webcam for the live scripts
- Model files kept in the project root:
  - `hand_landmarker.task` for the MediaPipe hand-landmark workflow
  - `yolov8n.pt` for YOLO-assisted processing
- A working installation of the packages pinned in `requirements.txt`

Ultralytics installs PyTorch and related packages, so leave sufficient disk space for the virtual environment and package downloads.

## Setup

In PowerShell, from the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation for the current session, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment again.

## Quick start

### Process a still image, video, or image folder

`gesture_input.py` runs without a webcam when an input path is supplied. It writes annotated media and batch summaries to `output/`.

```powershell
# One image
python gesture_input.py --image path\to\image.jpg

# One video
python gesture_input.py --video path\to\video.mp4

# Every supported image in a directory
python gesture_input.py --batch path\to\images
```

### Run the real-time gesture demo

```powershell
python live_gesture.py
```

Press `Q` in the OpenCV window to stop it. This script requires a webcam. Its drone-SDK examples are commented out, so it prints simulated command activity unless you deliberately add an integration.

## Gesture-command mapping

The runtime scripts classify finger-state patterns into these labels:

| Hand pose | Command label |
| --- | --- |
| Fist | `LAND` |
| Open palm | `HOVER` |
| Thumb only / index-front pattern | `TAKEOFF` |
| Index finger up | `ASCEND` |
| Pinky up | `DESCEND` |
| Peace sign | `FORWARD` |
| Four fingers | `EMERGENCY` |

The mapping is defined in `gesture_input.py` and `live_gesture.py`; adjust it to match the actions appropriate for your application.

## Dataset and training workflow

The classifier pipeline uses each hand's 21 landmarks as 63 features: all `x`, then `y`, then `z` coordinates, followed by a gesture label.

```text
labelled hand images
        │
        ▼
landmark_extraction.py ──► gesture_dataset.csv
        │
        ▼
augment_data.py ─────────► gesture_dataset_augmented.csv
        │
        ▼
train_classifier.py ─────► gesture_model.pkl
```

### 1. Extract landmarks

Update `BASE_PATH` in `landmark_extraction.py` to point to your labelled dataset. The script expects this layout:

```text
<dataset-root>/
  train/images/ and train/labels/
  valid/images/ and valid/labels/
  test/images/  and test/labels/
```

Labels are expected in YOLO text-file format. The script filters the configured gesture classes, detects a hand in each image, and creates `gesture_dataset.csv`.

```powershell
python landmark_extraction.py
```

### 2. Inspect and augment the dataset

```powershell
python check_data.py
python augment_data.py
```

`augment_data.py` adds small Gaussian noise to samples for the configured `strafe_left`, `strafe_right`, `forward`, and `land` classes. It creates `gesture_dataset_augmented.csv`; review or adapt `AUGMENT_TARGETS` before use.

### 3. Train the classifier

```powershell
python train_classifier.py
```

The training script reads `gesture_dataset.csv`, uses a stratified 80/20 train/test split, prints a classification report, and saves the fitted 100-tree Random Forest to `gesture_model.pkl`.

## Script guide

| Script | Purpose | Hardware needed |
| --- | --- | --- |
| `gesture_input.py` | Process an image, video, or image folder with landmark detection and annotation | No webcam for file modes |
| `live_gesture.py` | Show live gesture labels and simulated command execution | Webcam |
| `record_gesture.py` | Capture landmark samples for one configured gesture | Webcam |
| `landmark_extraction.py` | Extract labelled landmarks from an image dataset | No webcam |
| `augment_data.py` | Balance selected classes with noisy landmark samples | No webcam |
| `check_data.py` | Print label counts from `gesture_dataset.csv` | No webcam |
| `train_classifier.py` | Train and save the Random Forest model | No webcam |
| `test_gesture.py` | Basic MediaPipe webcam test | Webcam |
| `test_yolo.py` | Measure YOLO webcam performance | Webcam |
| `rename_images.py` | Rename images for data preparation | No webcam |

## Project files

```text
hand_landmarker.task          MediaPipe hand-landmarker model
gesture_recognizer.task       Additional MediaPipe gesture-recognizer asset
yolov8n.pt                    YOLO model used by the YOLO-assisted scripts
gesture_dataset.csv           Landmark dataset
gesture_dataset_augmented.csv Augmented landmark dataset
gesture_model.pkl             Trained Random Forest model
requirements.txt              Pinned direct Python dependencies
```

## Notes and limitations

- `landmark_extraction.py` contains a machine-specific dataset path and must be updated before use elsewhere.
- `record_gesture.py` writes to `gesture_dataset_augmented.csv`; make a backup before collecting additional samples.
- The generated `gesture_model.pkl` is not currently loaded by the real-time command scripts. Those scripts use rule-based finger-state matching.
- Performance and detection quality depend on lighting, camera quality, pose variation, and the supplied model files.

## License

No license file is currently included. Add an explicit license before redistributing or using this project as a dependency.
