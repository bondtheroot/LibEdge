import os
# Force Qt to use X11/XWayland to prevent Wayland display crashes on Linux
os.environ["QT_QPA_PLATFORM"] = "xcb"

import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO

import warnings
warnings.filterwarnings("ignore")

# Select the fastest CUDA convolution kernels
torch.backends.cudnn.benchmark = True

# ================= Directory & Path Setup =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VIDEO_SOURCE = os.path.join(BASE_DIR, "dataset", "raw", "library.mp4")
OUTPUT_PATH = os.path.join(BASE_DIR, "outputs", "occupancy_video.mp4")

POSE_MODEL_PATH = os.path.join(BASE_DIR, "src", "yolo11n-pose.pt")
DETECT_MODEL_PATH = os.path.join(BASE_DIR, "src", "yolo11n.pt")

# Fallback to root directory if model weights reside at repo root
if not os.path.exists(POSE_MODEL_PATH):
    POSE_MODEL_PATH = os.path.join(BASE_DIR, "yolo11n-pose.pt")
if not os.path.exists(DETECT_MODEL_PATH):
    DETECT_MODEL_PATH = os.path.join(BASE_DIR, "yolo11n.pt")

# ================= Configuration =================
PERSON_CONF = 0.25
CHAIR_CONF = 0.17    # Tuned to capture small/backlit rear chairs
CHAIR_CLASS = 56
# =================================================


def is_sitting_posture(keypoints, confidences):
    """Compares thigh vertical drop against torso length."""
    knee_conf = max(confidences[13], confidences[14])

    # Occlusion fallback: sitting behind an opaque desk
    if knee_conf < 0.25:
        return True

    hip_y = (keypoints[11][1] + keypoints[12][1]) / 2.0
    knee_y = (keypoints[13][1] + keypoints[14][1]) / 2.0
    shoulder_y = (keypoints[5][1] + keypoints[6][1]) / 2.0

    torso_len = abs(hip_y - shoulder_y)
    thigh_drop = abs(knee_y - hip_y)

    if torso_len > 0 and (thigh_drop / torso_len) < 0.85:
        return True
    return False


def box_contains_point(box, point):
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return (inter / union) if union > 0 else 0


class DynamicSpaceTracker:
    def analyze(self, pose_results, raw_chairs):
        # 1. Extract seated individuals using pose keypoints
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
                    if kp_c[11] > 0.3 or kp_c[12] > 0.3:
                        hip_x = float((kp[11][0] + kp[12][0]) / 2.0)
                        hip_y = float((kp[11][1] + kp[12][1]) / 2.0)
                    else:
                        hip_x = float((box[0] + box[2]) / 2.0)
                        hip_y = float(box[1] + (box[3] - box[1]) * 0.75)

                    seated_people.append({
                        "box": box,
                        "hip_point": (hip_x, hip_y)
                    })

        # 2. Identify vacant chairs (no seated person intersecting)
        vacant_chairs = []
        for cbox in raw_chairs:
            is_occupied = any(
                box_contains_point(cbox, p["hip_point"]) or compute_iou(cbox, p["box"]) > 0.15
                for p in seated_people
            )
            if not is_occupied:
                vacant_chairs.append(cbox)

        return seated_people, vacant_chairs


def render_display(frame, seated, vacant):
    overlay = frame.copy()

    # 1. Occupied Seats (Blue)
    for p in seated:
        b = p["box"]
        cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (52, 152, 219), 2)
        cv2.circle(frame, (int(p["hip_point"][0]), int(p["hip_point"][1])), 4, (0, 0, 255), -1)
        cv2.putText(frame, "OCCUPIED", (b[0], max(18, b[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (52, 152, 219), 2)

    # 2. Vacant Chairs (Green)
    for cbox in vacant:
        cv2.rectangle(overlay, (cbox[0], cbox[1]), (cbox[2], cbox[3]), (46, 204, 113), -1)
        cv2.rectangle(frame, (cbox[0], cbox[1]), (cbox[2], cbox[3]), (46, 204, 113), 2)
        cv2.putText(frame, "VACANT", (cbox[0], max(18, cbox[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (46, 204, 113), 1)

    # Top Analytics Dashboard Banner
    banner = f"Vacant: {len(vacant)}  |  Occupied: {len(seated)}"
    cv2.rectangle(frame, (8, 6), (320, 36), (20, 20, 20), -1)
    cv2.putText(frame, banner, (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

    return cv2.addWeighted(overlay, 0.20, frame, 0.80, 0)


def main():
    print(f"[*] Input stream: {VIDEO_SOURCE}")
    print(f"[*] Target output: {OUTPUT_PATH}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    pose_model = YOLO(POSE_MODEL_PATH)
    detect_model = YOLO(DETECT_MODEL_PATH)

    if device != "cpu":
        pose_model.to(device).to(torch.float16)
        detect_model.to(device).to(torch.float16)
    else:
        pose_model.to(device)
        detect_model.to(device)

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Error: Could not open {VIDEO_SOURCE}")
        return

    native_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    native_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (native_w, native_h))

    tracker = DynamicSpaceTracker()

    # ================= Performance Tuning =================
    FRAME_STRIDE = 2
    INFERENCE_IMGSZ = min(max(native_w, native_h), 960)
    # ======================================================

    t_start = time.time()
    frame_idx = 0

    seated, vacant = [], []

    print(f"[*] Processing {total_frames} frames [Device: {device}, FP16: {device != 'cpu'}, Stride: {FRAME_STRIDE}, ImgSz: {INFERENCE_IMGSZ}]...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Run dual YOLO inference on stride steps only
        if frame_idx % FRAME_STRIDE == 0 or frame_idx == 1:
            # 1. Pose estimation
            pose_results = pose_model.predict(
                frame, 
                imgsz=INFERENCE_IMGSZ, 
                conf=PERSON_CONF, 
                device=device,
                verbose=False
            )[0]

            # 2. Chair detection
            detect_results = detect_model.predict(
                frame, 
                imgsz=INFERENCE_IMGSZ, 
                classes=[CHAIR_CLASS], 
                conf=CHAIR_CONF, 
                device=device,
                verbose=False
            )[0]

            # 3. Class-specific filtering
            raw_chairs = []
            for box in detect_results.boxes:
                b = box.xyxy[0].cpu().numpy().astype(int)
                cls_id = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())

                if cls_id == CHAIR_CLASS and conf >= CHAIR_CONF:
                    raw_chairs.append(b)

            # 4. State updates
            seated, vacant = tracker.analyze(pose_results, raw_chairs)

        # Render overlays using current or cached detection states
        rendered = render_display(frame, seated, vacant)
        out.write(rendered)

        # Progress stats
        if frame_idx % 30 == 0 or frame_idx == total_frames:
            elapsed = time.time() - t_start
            cur_fps = frame_idx / elapsed if elapsed > 0 else 0
            eta = (total_frames - frame_idx) / cur_fps if cur_fps > 0 else 0
            pct = (frame_idx / total_frames) * 100
            print(
                f"\rProgress: {pct:5.1f}% | Frame {frame_idx}/{total_frames} | {cur_fps:.1f} FPS | ETA: {eta:.0f}s", 
                end="", 
                flush=True
            )

    print()
    cap.release()
    out.release()
    print(f"[✓] Processing complete. Video saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
