import os
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import mediapipe as mp
import cv2
import time
from datetime import datetime

# ── Drone SDK setup (uncomment when drone is available) ────────────
# DJI Tello:
#   from djitellopy import Tello
#   tello = Tello()
#   tello.connect()
#   tello.streamon()
#   print(f"Battery: {tello.get_battery()}%")
#
# ArduPilot/dronekit:
#   from dronekit import connect, VehicleMode
#   vehicle = connect('/dev/ttyUSB0', baud=57600, wait_ready=True)
#
# MAVLink/pymavlink:
#   from pymavlink import mavutil
#   master = mavutil.mavlink_connection('/dev/ttyUSB0', baud=57600)
#   master.wait_heartbeat()
#
# Custom/other:
#   import your_drone_sdk
#   drone = your_drone_sdk.connect()

# ── MediaPipe setup ────────────────────────────────────────────────
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# ── Finger state detector ──────────────────────────────────────────
def get_finger_states(landmarks, tolerance=0.02):
    fingers = []
    fingers.append(1 if landmarks[4].x < landmarks[3].x - tolerance else 0)
    for tip, dip in zip([8, 12, 16, 20], [6, 10, 14, 18]):
        diff = landmarks[dip].y - landmarks[tip].y
        fingers.append(1 if diff > tolerance else 0)
    return tuple(fingers)

# ── Gesture map ────────────────────────────────────────────────────
GESTURE_MAP = {
    (0, 0, 0, 0, 0): ("fist",         "LAND",      (0, 0, 255)),
    (1, 1, 1, 1, 1): ("open_palm",    "HOVER",     (255, 255, 0)),
    (1, 0, 0, 0, 0): ("index_front",  "TAKEOFF",   (0, 255, 0)),
    (0, 1, 0, 0, 0): ("point_up",     "ASCEND",    (0, 255, 255)),
    (0, 0, 0, 0, 1): ("pinky_up",     "DESCEND",   (255, 0, 255)),
    (0, 1, 1, 0, 0): ("peace",        "FORWARD",   (255, 165, 0)),
    (1, 1, 1, 1, 0): ("four_fingers", "EMERGENCY", (0, 0, 255)),
}

# ── Closest match classifier ───────────────────────────────────────
def classify_gesture(states):
    best_match = None
    best_score = -1
    for pattern, gesture_info in GESTURE_MAP.items():
        score = sum(1 for a, b in zip(states, pattern) if a == b)
        if score > best_score:
            best_score = score
            best_match = gesture_info
    return best_match if best_score >= 4 else None

# ── Command durations ──────────────────────────────────────────────
COMMAND_DURATION = {
    "TAKEOFF":   2.0,
    "LAND":      2.0,
    "HOVER":     0.0,
    "ASCEND":    1.0,
    "DESCEND":   1.0,
    "FORWARD":   1.5,
    "EMERGENCY": 0.0,
}

# ── Execute command ────────────────────────────────────────────────
def execute_command(command):
    duration = COMMAND_DURATION.get(command, 1.0)
    print(f"[DRONE] >> {command}", end="")
    if duration > 0:
        print(f" (for {duration}s)")
    else:
        print(" (instant)")

    # ── DRONE SDK COMMANDS (uncomment when drone is available) ──
    #
    # DJI Tello (djitellopy):
    # if command == "TAKEOFF":
    #     tello.takeoff()
    # elif command == "LAND":
    #     tello.land()
    # elif command == "HOVER":
    #     tello.send_rc_control(0, 0, 0, 0)
    # elif command == "ASCEND":
    #     tello.send_rc_control(0, 0, 20, 0)
    #     time.sleep(duration)
    #     tello.send_rc_control(0, 0, 0, 0)
    # elif command == "DESCEND":
    #     tello.send_rc_control(0, 0, -20, 0)
    #     time.sleep(duration)
    #     tello.send_rc_control(0, 0, 0, 0)
    # elif command == "FORWARD":
    #     tello.send_rc_control(0, 20, 0, 0)
    #     time.sleep(duration)
    #     tello.send_rc_control(0, 0, 0, 0)
    # elif command == "EMERGENCY":
    #     tello.emergency()
    #
    # ArduPilot/dronekit:
    # if command == "TAKEOFF":
    #     vehicle.simple_takeoff(1)
    # elif command == "LAND":
    #     vehicle.mode = VehicleMode("LAND")
    # elif command == "HOVER":
    #     vehicle.mode = VehicleMode("LOITER")
    # elif command == "EMERGENCY":
    #     vehicle.mode = VehicleMode("RTL")
    #
    # MAVLink/pymavlink:
    # if command == "TAKEOFF":
    #     master.mav.command_long_send(
    #         master.target_system, master.target_component,
    #         mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, 1)
    # elif command == "LAND":
    #     master.mav.command_long_send(
    #         master.target_system, master.target_component,
    #         mavutil.mavlink.MAV_CMD_NAV_LAND, 0, 0, 0, 0, 0, 0, 0, 0)
    # elif command == "EMERGENCY":
    #     master.mav.command_long_send(
    #         master.target_system, master.target_component,
    #         mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 0, 21196,
    #         0, 0, 0, 0, 0)

# ── Debounce state ─────────────────────────────────────────────────
def save_command_snapshot(frame, command_label):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join("output", f"{command_label}_{timestamp}.jpg")
    if cv2.imwrite(output_path, frame):
        print(f"[SNAPSHOT] Saved → {output_path}")
    else:
        print("[SNAPSHOT] Failed to save screenshot")


HOLD_TIME = 0.5
COOLDOWN  = 1.5

current_gesture    = None
gesture_start_time = None
triggered_gesture  = None
last_command_time  = {}
executing          = False

# ── Manual control keys ────────────────────────────────────────────
MANUAL_KEYS = {
    ord('w'): "MANUAL: FORWARD",
    ord('s'): "MANUAL: BACKWARD",
    ord('a'): "MANUAL: LEFT",
    ord('d'): "MANUAL: RIGHT",
    ord('r'): "MANUAL: ASCEND",
    ord('f'): "MANUAL: DESCEND",
    ord('t'): "MANUAL: TAKEOFF",
    ord('l'): "MANUAL: LAND",
}

os.makedirs("output", exist_ok=True)

cap = cv2.VideoCapture(0)
session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
session_video_path = os.path.join("output", f"session_{session_timestamp}.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(session_video_path, fourcc, 20, (640, 480))
print("Drone Gesture Control — Phase 1 Simulation")
print("Gestures active. Press Q to quit.\n")

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (640, 480))
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── Key input ──────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('e'):
            print("[DRONE] >> EMERGENCY (keyboard override)")
            # DJI Tello:     tello.emergency()
            # ArduPilot:     vehicle.mode = VehicleMode("RTL")
            # MAVLink:       master.mav.command_long_send(...)
            executing = False

        if key == ord('q'):
            break

        if key in MANUAL_KEYS:
            print(f"[DRONE] >> {MANUAL_KEYS[key]}")

        # ── MediaPipe detection ────────────────────────────────────
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = landmarker.detect(mp_image)

        gesture_name  = None
        command_label = None
        command_color = (255, 255, 255)
        states        = None
        match         = None

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]

            for lm in landmarks:
                cx = int(lm.x * 640)
                cy = int(lm.y * 480)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

            states = get_finger_states(landmarks)
            match  = classify_gesture(states)

            if match:
                gesture_name, command_label, command_color = match
                now = time.time()

                if gesture_name == current_gesture:
                    held = now - gesture_start_time
                    if held >= HOLD_TIME and not executing:
                        last = last_command_time.get(command_label, 0)
                        if now - last >= COOLDOWN:
                            triggered_gesture = (command_label, command_color)
                            execute_command(command_label)
                            snapshot_held = min(time.time() - gesture_start_time, HOLD_TIME)
                            snapshot_progress = int((snapshot_held / HOLD_TIME) * 300)
                            cv2.rectangle(frame, (10, 440), (310, 460), (50, 50, 50), -1)
                            cv2.rectangle(frame, (10, 440), (10 + snapshot_progress, 460),
                                         command_color, -1)
                            cv2.putText(frame, "Hold...", (320, 455),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                            cv2.putText(frame, f"Gesture: {gesture_name}",
                                       (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                       0.8, (255, 255, 255), 2)
                            cv2.putText(frame, f"COMMAND: {command_label}",
                                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                                       1.2, command_color, 3)
                            cv2.putText(frame, f"Fingers: {states}",
                                       (10, 130), cv2.FONT_HERSHEY_SIMPLEX,
                                       0.6, (150, 150, 150), 1)
                            cv2.putText(frame, "MODE: MANUAL  |  E=Emergency  Q=Quit",
                                       (10, 475), cv2.FONT_HERSHEY_SIMPLEX,
                                       0.5, (150, 150, 150), 1)
                            save_command_snapshot(frame, command_label)
                            last_command_time[command_label] = now
                            executing = True
                else:
                    current_gesture    = gesture_name
                    gesture_start_time = now
                    triggered_gesture  = None
                    executing          = False

                held     = min(time.time() - gesture_start_time, HOLD_TIME)
                progress = int((held / HOLD_TIME) * 300)
                cv2.rectangle(frame, (10, 440), (310, 460), (50, 50, 50), -1)
                cv2.rectangle(frame, (10, 440), (10 + progress, 460),
                             command_color, -1)
                cv2.putText(frame, "Hold...", (320, 455),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            else:
                current_gesture    = None
                gesture_start_time = None
                triggered_gesture  = None
                executing          = False

        else:
            current_gesture    = None
            gesture_start_time = None
            executing          = False

        # ── HUD ───────────────────────────────────────────────────
        if gesture_name:
            cv2.putText(frame, f"Gesture: {gesture_name}",
                       (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                       0.8, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "No gesture",
                       (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                       0.8, (0, 0, 255), 2)

        if triggered_gesture:
            label, color = triggered_gesture
            cv2.putText(frame, f"COMMAND: {label}",
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                       1.2, color, 3)

        if states:
            cv2.putText(frame, f"Fingers: {states}",
                       (10, 130), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (150, 150, 150), 1)

        cv2.putText(frame, "MODE: MANUAL  |  E=Emergency  Q=Quit",
                   (10, 475), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (150, 150, 150), 1)

        out.write(frame)
        cv2.imshow("Drone Gesture Control", frame)

cap.release()
out.release()
cv2.destroyAllWindows()

# ── Drone cleanup (uncomment when drone is available) ──────────────
# DJI Tello:
#   tello.streamoff()
#   tello.end()
#
# ArduPilot/dronekit:
#   vehicle.close()
#
# MAVLink/pymavlink:
#   master.close()

print("\nSession ended.")
print(f"[SESSION] Video saved → {session_video_path}")
