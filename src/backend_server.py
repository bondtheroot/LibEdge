import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

import asyncio
import json
import threading
import time
from typing import Dict, List

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from ultralytics import YOLO

# ================= Configuration & Paths =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_SOURCE = os.path.join(BASE_DIR, "dataset", "raw", "library_01.mp4")
POSE_MODEL_PATH = os.path.join(BASE_DIR, "src", "yolo11n-pose.pt")
DETECT_MODEL_PATH = os.path.join(BASE_DIR, "src", "yolo11n.pt")

CONF_THRESHOLD = 0.25
HOGGING_TIMEOUT_SEC = 20

CHAIR_CLASS = 56
ITEM_CLASSES = {24: "backpack", 26: "handbag", 63: "laptop", 73: "book"}
# =========================================================

app = FastAPI(title="LibEdge Occupancy Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SystemState:
    """Thread-safe state container for streaming and API responses."""
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_frame: np.ndarray = None
        self.counts = {"vacant": 0, "occupied": 0, "hogged": 0, "unattended": 0}
        self.fps = 0.0
        self.last_updated = 0.0


state = SystemState()


class ConnectionManager:
    """Manages active WebSocket dashboard clients."""
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass


manager = ConnectionManager()


def is_sitting_posture(keypoints, confidences):
    """Evaluates whether joint positions match a seated posture."""
    hip_conf = max(confidences[11], confidences[12])
    knee_conf = max(confidences[13], confidences[14])

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


def run_pipeline():
    """Background worker thread running YOLO inference loop."""
    print("[*] Initializing models in inference worker thread...")
    pose_model = YOLO(POSE_MODEL_PATH)
    detect_model = YOLO(DETECT_MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    item_timers = {}
    target_classes = [CHAIR_CLASS] + list(ITEM_CLASSES.keys())

    t_prev = time.time()

    while True:
        if not cap.isOpened():
            cap = cv2.VideoCapture(VIDEO_SOURCE)

        ret, frame = cap.read()
        if not ret:
            # Loop video file for continuous edge simulation
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        now = time.time()
        fps_calc = 1.0 / (now - t_prev) if (now - t_prev) > 0 else 30.0
        t_prev = now

        # Run models
        pose_res = pose_model.predict(frame, conf=CONF_THRESHOLD, verbose=False)[0]
        detect_res = detect_model.predict(frame, classes=target_classes, conf=CONF_THRESHOLD, verbose=False)[0]

        # 1. Seated people
        seated_people = []
        if pose_res.keypoints is not None:
            boxes = pose_res.boxes.xyxy.cpu().numpy()
            kpts = pose_res.keypoints.xy.cpu().numpy()
            kpt_confs = pose_res.keypoints.conf.cpu().numpy()

            for i in range(len(boxes)):
                b = boxes[i].astype(int)
                if is_sitting_posture(kpts[i], kpt_confs[i]):
                    hip_x = float((kpts[i][11][0] + kpts[i][12][0]) / 2.0)
                    hip_y = float((kpts[i][11][1] + kpts[i][12][1]) / 2.0)
                    seated_people.append({"box": b, "hip": (hip_x, hip_y)})

        # 2. Chairs & Items
        chairs = []
        items = []
        for box in detect_res.boxes:
            b = box.xyxy[0].cpu().numpy().astype(int)
            cls_id = int(box.cls[0].cpu().numpy())
            if cls_id == CHAIR_CLASS:
                chairs.append(b)
            elif cls_id in ITEM_CLASSES:
                items.append(b)

        # 3. Vacancy evaluation
        vacant_chairs = []
        for c in chairs:
            occupied = False
            for p in seated_people:
                hx, hy = p["hip"]
                if (c[0] <= hx <= c[2] and c[1] <= hy <= c[3]):
                    occupied = True
                    break
            if not occupied:
                vacant_chairs.append(c)

        # 4. Hogging evaluation
        hogged_items = []
        unattended_items = []
        current_keys = set()

        for it in items:
            cx, cy = (it[0] + it[2]) / 2.0, (it[1] + it[3]) / 2.0
            attended = any(np.hypot(cx - p["hip"][0], cy - p["hip"][1]) < 120 for p in seated_people)

            if not attended:
                slot_key = f"{int(cx // 30)}_{int(cy // 30)}"
                current_keys.add(slot_key)
                if slot_key not in item_timers:
                    item_timers[slot_key] = now

                elapsed = now - item_timers[slot_key]
                if elapsed >= HOGGING_TIMEOUT_SEC:
                    hogged_items.append(it)
                else:
                    unattended_items.append(it)

        # Clean stale timers
        for k in [k for k in item_timers if k not in current_keys]:
            del item_timers[k]

        # Draw visual HUD on frame
        annotated = frame.copy()
        for p in seated_people:
            cv2.rectangle(annotated, (p["box"][0], p["box"][1]), (p["box"][2], p["box"][3]), (52, 152, 219), 2)
        for c in vacant_chairs:
            cv2.rectangle(annotated, (c[0], c[1]), (c[2], c[3]), (46, 204, 113), 2)
        for h in hogged_items:
            cv2.rectangle(annotated, (h[0], h[1]), (h[2], h[3]), (231, 76, 60), 2)

        # Update global state under thread lock
        with state.lock:
            state.latest_frame = annotated
            state.fps = round(fps_calc, 1)
            state.last_updated = now
            state.counts = {
                "vacant": len(vacant_chairs),
                "occupied": len(seated_people),
                "hogged": len(hogged_items),
                "unattended": len(unattended_items)
            }

        time.sleep(0.01)


@app.on_event("startup")
def startup_event():
    """Starts the video analytics engine in a daemon thread on server boot."""
    thread = threading.Thread(target=run_pipeline, daemon=True)
    thread.start()


# ================= REST API Endpoints =================

@app.get("/api/health")
def health_check():
    return {"status": "healthy", "service": "LibEdge CV Engine"}


@app.get("/api/occupancy")
def get_occupancy():
    with state.lock:
        return JSONResponse({
            "status": "success",
            "timestamp": state.last_updated,
            "metrics": {
                "vacant_seats": state.counts["vacant"],
                "occupied_seats": state.counts["occupied"],
                "hogged_seats": state.counts["hogged"],
                "unattended_items": state.counts["unattended"],
                "total_capacity": state.counts["vacant"] + state.counts["occupied"] + state.counts["hogged"]
            },
            "performance": {"fps": state.fps}
        })


# ================= Live Video MJPEG Stream =================

def mjpeg_generator():
    """Encodes frames to JPEG on-the-fly for real-time web browser preview."""
    while True:
        with state.lock:
            if state.latest_frame is None:
                time.sleep(0.05)
                continue
            ret, jpeg = cv2.imencode(".jpg", state.latest_frame)
            if not ret:
                continue
            frame_bytes = jpeg.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
        time.sleep(0.03)


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# ================= Real-Time WebSocket Channel =================

@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            with state.lock:
                payload = {
                    "timestamp": state.last_updated,
                    "counts": state.counts,
                    "fps": state.fps
                }
            await websocket.send_json(payload)
            await asyncio.sleep(0.2)  # 5 Hz push rate
    except WebSocketDisconnect:
        manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend_server:app", host="0.0.0.0", port=8000, reload=False)
