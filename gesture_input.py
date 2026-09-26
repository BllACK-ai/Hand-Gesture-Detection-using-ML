import os
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import mediapipe as mp
import cv2
import argparse
import csv
from datetime import datetime
from ultralytics import YOLO


yolo_model = YOLO("yolov8n.pt")

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)


def select_gesture_hand(hand_landmarks_list):
    """
    Given a list of detected hand landmark sets, returns the one
    whose wrist landmark (landmark 0) has the smallest y value,
    meaning it is highest in the frame - most likely the raised
    gesture hand.
    """
    return min(hand_landmarks_list, key=lambda hand: hand[0].y)


def get_finger_states(landmarks, tolerance=0.02):
    fingers = []
    fingers.append(1 if landmarks[4].x < landmarks[3].x - tolerance else 0)
    for tip, dip in zip([8, 12, 16, 20], [6, 10, 14, 18]):
        diff = landmarks[dip].y - landmarks[tip].y
        fingers.append(1 if diff > tolerance else 0)
    return tuple(fingers)


GESTURE_MAP = {
    (0, 0, 0, 0, 0): ("fist",         "LAND",      (0, 0, 255)),
    (1, 1, 1, 1, 1): ("open_palm",    "HOVER",     (255, 255, 0)),
    (1, 0, 0, 0, 0): ("index_front",  "TAKEOFF",   (0, 255, 0)),
    (0, 1, 0, 0, 0): ("point_up",     "ASCEND",    (0, 255, 255)),
    (0, 0, 0, 0, 1): ("pinky_up",     "DESCEND",   (255, 0, 255)),
    (0, 1, 1, 0, 0): ("peace",        "FORWARD",   (255, 165, 0)),
    (1, 1, 1, 1, 0): ("four_fingers", "EMERGENCY", (0, 0, 255)),
}


def classify_gesture(states):
    best_match = None
    best_score = -1
    for pattern, gesture_info in GESTURE_MAP.items():
        score = sum(1 for a, b in zip(states, pattern) if a == b)
        if score > best_score:
            best_score = score
            best_match = gesture_info
    return best_match if best_score >= 4 else None


def draw_landmarks(frame, landmarks):
    for lm in landmarks:
        cx = int(lm.x * 640)
        cy = int(lm.y * 480)
        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)


def draw_remapped_landmarks(frame, landmarks):
    h, w = frame.shape[:2]
    for x, y in landmarks:
        cx = int(x * w)
        cy = int(y * h)
        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)


def draw_gesture_text(frame, gesture_name, command_label, command_color, states):
    cv2.putText(
        frame,
        f"Gesture: {gesture_name}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )
    cv2.putText(
        frame,
        f"Command: {command_label}",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        command_color,
        2
    )
    cv2.putText(
        frame,
        f"Fingers: {states}",
        (10, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (150, 150, 150),
        1
    )


def draw_no_hand(frame):
    cv2.putText(
        frame,
        "No hand detected",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )


def extract_hand_roi(frame, yolo_model, padding=80):
    """
    Uses YOLO to detect a person in the frame, crops the upper body
    region where the hand is likely to be, and returns the crop
    along with the offset coordinates for landmark remapping.
    Returns (roi, x_offset, y_offset) if person detected.
    Returns (original_frame, 0, 0) if no person detected.
    """
    h, w = frame.shape[:2]
    results = yolo_model(frame, classes=[0], verbose=False)

    if not results[0].boxes:
        print("[YOLO] No person detected - using full frame")
        return frame, 0, 0

    boxes = results[0].boxes
    box_index = int(boxes.conf.argmax().item()) if boxes.conf is not None else 0
    box = boxes[box_index]
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    # Extend upward to capture hands raised above the head
    y1 = max(0, y1 - int((y2 - y1) * 0.3))
    # Trim lower legs region only, keep most of the bounding box
    y2 = y1 + int((y2 - y1) * 0.85)

    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        print("[YOLO] No person detected - using full frame")
        return frame, 0, 0

    print("[YOLO] Person detected - cropping ROI for MediaPipe")
    return roi, x1, y1


def remap_landmarks(landmarks, roi_w, roi_h, x_offset, y_offset, full_w, full_h):
    remapped = []
    for lm in landmarks:
        full_x = (lm.x * roi_w + x_offset) / full_w
        full_y = (lm.y * roi_h + y_offset) / full_h
        remapped.append((full_x, full_y))
    return remapped


def detect_and_annotate_with_roi(frame, landmarker):
    roi, x_offset, y_offset = extract_hand_roi(frame, yolo_model)
    full_h, full_w = frame.shape[:2]
    roi_h, roi_w = roi.shape[:2]
    roi_used = roi_w != full_w or roi_h != full_h or x_offset != 0 or y_offset != 0

    if roi_used:
        x1, y1 = x_offset, y_offset
        x2, y2 = x_offset + roi_w, y_offset + roi_h
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            "ROI",
            (x1, max(15, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        draw_no_hand(frame)
        return None, None, None

    landmarks = select_gesture_hand(result.hand_landmarks)
    remapped = remap_landmarks(
        landmarks,
        roi_w,
        roi_h,
        x_offset,
        y_offset,
        full_w,
        full_h
    )

    states = get_finger_states(landmarks)
    match = classify_gesture(states)

    if match:
        gesture_name, command_label, command_color = match
    else:
        gesture_name, command_label, command_color = "unknown", "NONE", (0, 0, 255)

    draw_gesture_text(frame, gesture_name, command_label, command_color, states)
    return gesture_name, command_label, states


def detect_and_annotate(frame, landmarker):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        draw_no_hand(frame)
        return None, None, None

    landmarks = result.hand_landmarks[0]
    draw_landmarks(frame, landmarks)

    states = get_finger_states(landmarks)
    match = classify_gesture(states)

    if match:
        gesture_name, command_label, command_color = match
    else:
        gesture_name, command_label, command_color = "unknown", "NONE", (0, 0, 255)

    draw_gesture_text(frame, gesture_name, command_label, command_color, states)
    return gesture_name, command_label, states


def process_image(image_path):
    if not os.path.exists(image_path):
        print(f"[ERROR] Image file does not exist: {image_path}")
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Could not load image: {image_path}")
        return

    frame = cv2.flip(image, 1)
    frame = cv2.resize(frame, (640, 480))

    with HandLandmarker.create_from_options(options) as landmarker:
        gesture_name, command_label, _ = detect_and_annotate_with_roi(frame, landmarker)

    os.makedirs("output", exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join("output", f"{base_name}_{timestamp}.jpg")
    if cv2.imwrite(output_path, frame):
        print(f"[IMAGE] Saved → {output_path}")
    else:
        print("[ERROR] Failed to save output image")

    if gesture_name and command_label:
        print(f"[IMAGE] gesture: {gesture_name} -> {command_label}")
    else:
        print("[IMAGE] gesture: none -> NONE")

    cv2.imshow("Gesture Image", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_batch(folder_path):
    if not os.path.isdir(folder_path):
        print(f"[ERROR] Batch folder does not exist: {folder_path}")
        return

    supported_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = [
        os.path.join(folder_path, filename)
        for filename in sorted(os.listdir(folder_path))
        if os.path.splitext(filename)[1].lower() in supported_extensions
    ]

    total = len(image_paths)
    if total == 0:
        print(f"[BATCH] No supported image files found in {folder_path}")
        return

    print(f"[BATCH] Found {total} images in {folder_path}")

    os.makedirs("output", exist_ok=True)
    batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_rows = []
    detected_count = 0

    with HandLandmarker.create_from_options(options) as landmarker:
        for index, image_path in enumerate(image_paths, start=1):
            filename = os.path.basename(image_path)
            image = cv2.imread(image_path)

            if image is None:
                print(f"[BATCH] Skipped {filename} - could not load image")
                continue

            frame = cv2.flip(image, 1)
            frame = cv2.resize(frame, (640, 480))
            gesture_name, command_label, states = detect_and_annotate_with_roi(frame, landmarker)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            original_filename = os.path.basename(image_path)
            output_path = os.path.join("output", f"{original_filename}_{timestamp}.jpg")
            cv2.imwrite(output_path, frame)

            if gesture_name and command_label:
                detected_count += 1
                summary_rows.append({
                    "filename": filename,
                    "gesture_detected": gesture_name,
                    "command": command_label,
                    "finger_states": states,
                })
                print(
                    f"[BATCH] {index}/{total} - {filename} - "
                    f"gesture: {gesture_name} -> {command_label}"
                )
            else:
                summary_rows.append({
                    "filename": filename,
                    "gesture_detected": "none",
                    "command": "none",
                    "finger_states": "none",
                })
                print(f"[BATCH] {index}/{total} - {filename} - No hand detected")

    summary_path = os.path.join("output", f"batch_summary_{batch_timestamp}.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["filename", "gesture_detected", "command", "finger_states"]
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"[BATCH] Complete - {detected_count}/{total} gestures detected")
    print("[BATCH] Results saved -> output/")


def process_video(video_path):
    if not os.path.exists(video_path):
        print(f"[ERROR] Video file does not exist: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 30

    os.makedirs("output", exist_ok=True)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join("output", f"{base_name}_{timestamp}.mp4")

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (640, 480)
    )

    frame_number = 0
    processed = 0
    detected_count = 0
    latest_gesture = None

    with HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_number += 1

            if frame_number % 10 == 0:
                processed += 1
                frame = cv2.flip(frame, 1)
                frame = cv2.resize(frame, (640, 480))

                try:
                    gesture_name, _, _ = detect_and_annotate(frame, landmarker)
                    if gesture_name:
                        latest_gesture = gesture_name
                        detected_count += 1
                except Exception:
                    pass
            else:
                frame = cv2.resize(frame, (640, 480))

            writer.write(frame)

            if frame_number % 50 == 0:
                print(f"[VIDEO] frame {frame_number}/{total} - gesture: {latest_gesture}")

    cap.release()
    writer.release()

    print(f"[VIDEO] Done. Saved → {output_path}")
    print(f"[VIDEO] Gestures detected: {detected_count} frames out of {processed} processed")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run gesture recognition on an image or video file."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--image", help="Path to an input image")
    group.add_argument("--video", help="Path to an input video")
    group.add_argument("--batch", help="Path to a folder of input images")
    return parser, parser.parse_args()


def main():
    parser, args = parse_args()

    if not args.image and not args.video and not args.batch:
        parser.print_usage()
        return

    if args.image:
        process_image(args.image)
    elif args.video:
        process_video(args.video)
    elif args.batch:
        process_batch(args.batch)


if __name__ == "__main__":
    main()
