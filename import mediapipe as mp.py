import mediapipe as mp
import cv2

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path="gesture_recognizer.task"),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3
)

cap = cv2.VideoCapture(0)

with GestureRecognizer.create_from_options(options) as recognizer:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)          # mirror fix
        frame = cv2.resize(frame, (640, 480))  # better resolution

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = recognizer.recognize(mp_image)

        if result.gestures:
            gesture = result.gestures[0][0].category_name
            score = result.gestures[0][0].score
            color = (0, 255, 0) if score > 0.5 else (0, 165, 255)
            cv2.putText(frame, f"{gesture} ({score:.2f})",
                       (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                       1.0, color, 2)
        else:
            cv2.putText(frame, "No gesture",
                       (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                       1.0, (0, 0, 255), 2)

        cv2.imshow("Gesture Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()