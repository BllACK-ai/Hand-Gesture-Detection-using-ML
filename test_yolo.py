import cv2
from ultralytics import YOLO
import time

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

fps_list = []
prev_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # add this
    if not ret:
        break

    frame = cv2.resize(frame, (320, 240))
    results = model(frame, classes=[0], verbose=False)

    # FPS calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    fps_list.append(fps)

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Frames: {len(fps_list)}/100", (10, 45),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("YOLO FPS Test", frame)

    if len(fps_list) >= 100:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

avg_fps = sum(fps_list) / len(fps_list)
print(f"\nAverage YOLO FPS: {avg_fps:.1f}")
print(f"Recommended skip interval: every {max(3, int(30/avg_fps))} frames")