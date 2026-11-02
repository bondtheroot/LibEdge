import os
# Force Qt to use X11/XWayland to prevent Wayland crashes on Linux
os.environ["QT_QPA_PLATFORM"] = "xcb"

import time
import cv2
import numpy as np
from ultralytics import YOLO

# ================= Configuration =================
POSE_MODEL_NAME = "yolo11n-pose.pt"
DETECT_MODEL_NAME = "yolo11n.pt"

VIDEO_SOURCE = "/home/bond/LibEdge/dataset/raw/library_01.mp4"
OUTPUT_PATH = "./output.mp4"

CONF_THRESHOLD = 0.25
HOGGING_TIMEOUT_SEC = 20  # Set to 1800 for 30 minutes in production

# COCO Class IDs for detection model
CHAIR_CLASS = 56
ITEM_CLASSES = {24: "backpack", 26: "handbag", 63: "laptop", 73: "book"}

# COCO Keypoint IDs for pose model:
# 5: Left Shoulder, 6: Right Shoulder
# 11: Left Hip, 12: Right Hip
# 13: Left Knee, 14: Right Knee
# =================================================


def is_sitting_posture(keypoints, confidences):
    """
    Determines if a detected person is sitting based on 
    torso-to-thigh geometric relationships.
    """
    # Check hips and knees availability
    hip_conf = max(confidences[11], confidences[12])
    knee_conf = max(confidences[13], confidences[14])
    shoulder_conf = max(confidences[5], confidences[6])

    # Default fallback: if lower body is completely occluded by a desk,
    # the person is sitting behind a desk
    if knee_conf < 0.25:
        return True

    # Midpoints
    hip_y = (keypoints[11][1] + keypoints[12][1]) / 2.0
    knee_y = (keypoints[13][1] + keypoints[14][1]) / 2.0
    shoulder_y = (keypoints[5][1] + keypoints[6][1]) / 2.0

    torso_len = abs(hip_y - shoulder_y)
    thigh_vertical_drop = abs(knee_y - hip_y)

    # When standing, thigh vertical drop is comparable to or greater than torso length.
    # When seated, thigh is horizontal, dropping vertical distance significantly.
    if torso_len > 0 and (thigh_vertical_drop / torso_len) < 0.85:
        return True

    return False


def box_contains_point(box, point):
    """Checks if point (px, py) is inside box [x1, y1, x2, y2]."""
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def compute_iou(boxA, boxB):
    """Calculates Intersection-over-Union."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter_area

    return inter_area / union if union > 0 else 0


class DynamicSpaceTracker:
    def __init__(self, hogging_timeout=HOGGING_TIMEOUT_SEC):
        self.hogging_timeout = hogging_timeout
        self.item_timers = {}

    def analyze(self, pose_results, detect_results):
        now = time.time()

        # 1. Extract seated persons
        seated_people = []
        if pose_results.keypoints is not None:
            boxes = pose_results.boxes.xyxy.cpu().numpy()
            kpts = pose_results.keypoints.xy.cpu().numpy()
            kpt_confs = pose_results.keypoints.conf.cpu().numpy()

            for i in range(len(boxes)):
                box = boxes[i].astype(int)
                kp = kpts[i]
                kp_c = kpt_confs[i]

                if is_sitting_posture(kp, kp_c):
                    # Hip center point
                    hip_x = float((kp[11][0] + kp[12][0]) / 2.0) if (kp_c[11] > 0.3 or kp_c[12] > 0.3) else float((box[0] + box[2]) / 2.0)
                    hip_y = float((kp[11][1] + kp[12][1]) / 2.0) if (kp_c[11] > 0.3 or kp_c[12] > 0.3) else float(box[1] + (box[3] - box[1]) * 0.75)
                    seated_people.append({
                        "box": box,
                        "hip_point": (hip_x, hip_y)
                    })

        # 2. Extract detected chairs and belongings
        raw_chairs = []
        raw_items = []

        for box in detect_results.boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            cls_id = int(box.cls[0].cpu().numpy())

            if cls_id == CHAIR_CLASS:
                raw_chairs.append(xyxy)
            elif cls_id in ITEM_CLASSES:
                raw_items.append(xyxy)

        # 3. Associate chairs with sitting people
        vacant_chairs = []
        for cbox in raw_chairs:
            is_occupied = False
            for p in seated_people:
                # Check if person hip point or box overlaps with this chair
                if box_contains_point(cbox, p["hip_point"]) or compute_iou(cbox, p["box"]) > 0.15:
                    is_occupied = True
                    break
            if not is_occupied:
                vacant_chairs.append(cbox)

        # 4. Filter unattended items & track seat-hogging
        hogged_items = []
        unattended_items = []
        current_item_keys = set()

        for it_box in raw_items:
            it_center = ((it_box[0] + it_box[2]) / 2.0, (it_box[1] + it_box[3]) / 2.0)
            
            # Check proximity to any seated student
            near_person = False
            for p in seated_people:
                dist = np.hypot(it_center[0] - p["hip_point"][0], it_center[1] - p["hip_point"][1])
                if dist < 120:  # Within 120 pixels is considered attended
                    near_person = True
                    break

            if not near_person:
                # Track persistence using spatial grid hash
                slot_key = f"{int(it_center[0] // 30)}_{int(it_center[1] // 30)}"
                current_item_keys.add(slot_key)

                if slot_key not in self.item_timers:
                    self.item_timers[slot_key] = now

                elapsed = now - self.item_timers[slot_key]
                if elapsed >= self.hogging_timeout:
                    hogged_items.append({"box": it_box, "elapsed": elapsed})
                else:
                    unattended_items.append({"box": it_box, "elapsed": elapsed})

        # Evict stale timers
        for stale in [k for k in self.item_timers if k not in current_item_keys]:
            del self.item_timers[stale]

        return seated_people, vacant_chairs, hogged_items, unattended_items


def render_display(frame, seated_people, vacant_chairs, hogged_items, unattended_items):
    overlay = frame.copy()

    # 1. Draw Occupied Seats (Blue)
    for p in seated_people:
        box = p["box"]
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (52, 152, 219), 2)
        cv2.circle(frame, (int(p["hip_point"][0]), int(p["hip_point"][1])), 4, (0, 0, 255), -1)
        cv2.putText(frame, "OCCUPIED SEAT", (box[0], max(18, box[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (52, 152, 219), 2)

    # 2. Draw Vacant Chairs (Green)
    for cbox in vacant_chairs:
        cv2.rectangle(overlay, (cbox[0], cbox[1]), (cbox[2], cbox[3]), (46, 204, 113), -1)
        cv2.rectangle(frame, (cbox[0], cbox[1]), (cbox[2], cbox[3]), (46, 204, 113), 2)
        cv2.putText(frame, "VACANT SEAT", (cbox[0], max(18, cbox[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (46, 204, 113), 1)

    # 3. Draw Hogged Seats (Red)
    for h in hogged_items:
        box = h["box"]
        elapsed = h["elapsed"]
        cv2.rectangle(overlay, (box[0], box[1]), (box[2], box[3]), (231, 76, 60), -1)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (231, 76, 60), 2)
        cv2.putText(frame, f"HOGGED SEAT ({int(elapsed)}s)", (box[0], max(18, box[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (231, 76, 60), 2)

    # 4. Draw Pending Unattended Items (Yellow)
    for u in unattended_items:
        box = u["box"]
        elapsed = u["elapsed"]
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (241, 196, 15), 2)
        cv2.putText(frame, f"UNATTENDED ({int(elapsed)}s)", (box[0], max(18, box[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (241, 196, 15), 1)

    # Summary metrics header
    occ_count = len(seated_people)
    vac_count = len(vacant_chairs)
    hog_count = len(hogged_items)

    banner = f"Vacant Seats: {vac_count}  |  Occupied Seats: {occ_count}  |  Hogged Seats: {hog_count}"
    cv2.rectangle(frame, (10, 8), (560, 36), (20, 20, 20), -1)
    cv2.putText(frame, banner, (18, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    return cv2.addWeighted(overlay, 0.20, frame, 0.80, 0)


def main():
    print("[*] Loading YOLO-Pose and YOLO-Detect models...")
    pose_model = YOLO(POSE_MODEL_NAME)
    detect_model = YOLO(DETECT_MODEL_NAME)

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Error: Could not open {VIDEO_SOURCE}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    tracker = DynamicSpaceTracker(hogging_timeout=HOGGING_TIMEOUT_SEC)
    detect_target_classes = [CHAIR_CLASS] + list(ITEM_CLASSES.keys())

    print(f"[*] Autonomous tracking active -> {OUTPUT_PATH}. Press 'q' to stop.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Pose estimation for people
        pose_results = pose_model.predict(frame, conf=CONF_THRESHOLD, verbose=False)[0]

        # 2. Object detection for chairs and belongings
        detect_results = detect_model.predict(frame, classes=detect_target_classes, conf=CONF_THRESHOLD, verbose=False)[0]

        # 3. Dynamic evaluation without any hardcoded zones
        seated, vacant, hogged, unattended = tracker.analyze(pose_results, detect_results)

        # 4. Render and record
        rendered = render_display(frame, seated, vacant, hogged, unattended)
        out.write(rendered)

        cv2.imshow("Zero-Calibration Seat Monitor", rendered)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"[✓] Saved resultant video to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
