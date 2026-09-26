import mediapipe as mp
import cv2
import csv
import os

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

CLASS_NAMES = [
    'Call-Me', 'Crossed-Finger', 'Hand-Raised', 'Hand-Spread',
    'Heart', 'Italian', 'Left-Fist', 'Long-Live-and-Prosper',
    'Love-You', 'Middle', 'Ok', 'Peace', 'Pinch', 'Point-Down',
    'Point-Left', 'Point-Right', 'Point-Up', 'Pray', 'Punch',
    'Right-Fist', 'Rised-Fist', 'Rock', 'Thumb-Down', 'Thumb-Up'
]

GESTURE_MAP = {
    'Thumb-Up':    'takeoff',
    'Thumb-Down':  'land',
    'Hand-Spread': 'hover',
    'Point-Up':    'ascend',
    'Point-Down':  'descend',
    'Point-Left':  'strafe_left',
    'Point-Right': 'strafe_right',
    'Peace':       'forward',
    'Punch':       'emergency',   # changed from Hand-Raised
}

BASE_PATH = r"C:\Users\dell\Documents\hands\HandGesture.v2i.yolov8"
SPLITS = ["train", "valid", "test"]

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3
)

rows = []
processed = 0
skipped = 0

with HandLandmarker.create_from_options(options) as landmarker:
    for split in SPLITS:
        images_dir = os.path.join(BASE_PATH, split, "images")
        labels_dir = os.path.join(BASE_PATH, split, "labels")

        if not os.path.exists(images_dir):
            print(f"Skipping {split} — folder not found")
            continue

        image_files = [f for f in os.listdir(images_dir)
                       if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        print(f"\nProcessing {split}: {len(image_files)} images")

        for filename in image_files:
            label_filename = os.path.splitext(filename)[0] + ".txt"
            label_path = os.path.join(labels_dir, label_filename)

            if not os.path.exists(label_path):
                skipped += 1
                continue

            with open(label_path, "r") as f:
                lines = f.readlines()

            if not lines:
                skipped += 1
                continue

            class_id = int(lines[0].split()[0])
            class_name = CLASS_NAMES[class_id]

            if class_name not in GESTURE_MAP:
                skipped += 1
                continue

            drone_command = GESTURE_MAP[class_name]

            img_path = os.path.join(images_dir, filename)
            image = cv2.imread(img_path)
            if image is None:
                skipped += 1
                continue

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                row = [lm.x for lm in landmarks] + \
                      [lm.y for lm in landmarks] + \
                      [lm.z for lm in landmarks]
                row.append(drone_command)
                rows.append(row)
                processed += 1
            else:
                skipped += 1

            if (processed + skipped) % 50 == 0:
                print(f"  Extracted: {processed} | Skipped: {skipped}", end="\r")

print(f"\n\nDone — Extracted: {processed} | Skipped: {skipped}")

with open("gesture_dataset.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(rows)

print("Saved → gesture_dataset.csv")