import mediapipe as mp
import cv2
import csv
import time

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3
)

GESTURE_NAME = "strafe_Right"  # change to strafe_right for next run
TARGET_SAMPLES = 80

rows = []
count = 0
cap = cv2.VideoCapture(0)

print(f"Recording: {GESTURE_NAME}")
print("Hold the gesture in front of camera. Press S to start, Q to quit.")

recording = False

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (640, 480))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        status = f"Samples: {count}/{TARGET_SAMPLES}"
        color = (0, 255, 0) if recording else (0, 0, 255)
        cv2.putText(frame, status, (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(frame, f"Gesture: {GESTURE_NAME}", (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if not recording:
            cv2.putText(frame, "Press S to start recording", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        if recording and result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            row = [lm.x for lm in landmarks] + \
                  [lm.y for lm in landmarks] + \
                  [lm.z for lm in landmarks]
            row.append(GESTURE_NAME)
            rows.append(row)
            count += 1

            if count >= TARGET_SAMPLES:
                print(f"Done! Collected {count} samples.")
                break

        cv2.imshow("Record Gesture", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            recording = True
            print("Recording started...")
        elif key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# Append to existing dataset
with open("gesture_dataset_augmented.csv", "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(rows)

print(f"Appended {len(rows)} samples to gesture_dataset_augmented.csv")