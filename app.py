import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

import time
import json
import asyncio
import cv2
import numpy as np
import torch
import pymongo
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from ultralytics import YOLO

import warnings
warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True

# ================= Setup & Models =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

# MongoDB Setup
MONGO_URL = os.environ.get("MONGO_URL", "mongodb://localhost:27017/")
mongo_client = pymongo.MongoClient(MONGO_URL)
db = mongo_client["libedge_db"]
cameras_collection = db["cameras"]

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

POSE_PATH = os.path.join(BASE_DIR, "src", "yolo11n-pose.pt")
DETECT_PATH = os.path.join(BASE_DIR, "src", "yolo11n.pt")
if not os.path.exists(POSE_PATH):
    POSE_PATH = os.path.join(BASE_DIR, "yolo11n-pose.pt")
if not os.path.exists(DETECT_PATH):
    DETECT_PATH = os.path.join(BASE_DIR, "yolo11n.pt")

print(f"[*] Starting LibEdge Vision Models on {DEVICE}...")
pose_model = YOLO(POSE_PATH)
detect_model = YOLO(DETECT_PATH)

if DEVICE != "cpu":
    pose_model.to(DEVICE).to(torch.float16)
    detect_model.to(DEVICE).to(torch.float16)

templates = Jinja2Templates(directory=TEMPLATES_DIR)
CAMERAS = []


def load_internal_store():
    global CAMERAS
    try:
        CAMERAS = list(cameras_collection.find({}, {"_id": False}))
    except Exception:
        CAMERAS = []
        
    if not CAMERAS:
        # Default seed entry with conf_threshold
        CAMERAS = [
            {
                "id": "cam_01",
                "name": "Main Reading Hall",
                "floor": "Floor 1",
                "zone": "Quiet Zone A",
                "source": "dataset/raw/library.mp4",
                "interval_sec": 30,
                "conf_threshold": 0.25,
                "total_seats": 0,
                "occupied": 0,
                "vacant": 0,
                "last_tested": "Never",
                "status": "Initialized"
            }
        ]
        persist_store()


def persist_store():
    for cam in CAMERAS:
        cameras_collection.update_one(
            {"id": cam["id"]},
            {"$set": cam},
            upsert=True
        )


# ================= Dynamic Geometry & Association =================
def is_sitting_posture(keypoints, confidences):
    knee_conf = max(confidences[13], confidences[14])
    if knee_conf < 0.25:
        return True
    hip_y = (keypoints[11][1] + keypoints[12][1]) / 2.0
    knee_y = (keypoints[13][1] + keypoints[14][1]) / 2.0
    shoulder_y = (keypoints[5][1] + keypoints[6][1]) / 2.0
    torso_len = abs(hip_y - shoulder_y)
    thigh_drop = abs(knee_y - hip_y)
    return torso_len > 0 and (thigh_drop / torso_len) < 0.85


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


def analyze_frame_dynamically(frame, conf_threshold=0.25):
    """Detects chairs and sitting people dynamically from the frame using a given confidence threshold."""
    h, w = frame.shape[:2]
    imgsz = min(max(w, h), 960)

    # 1. Pose estimation for people
    pose_res = pose_model.predict(frame, imgsz=imgsz, conf=conf_threshold, device=DEVICE, verbose=False)[0]
    seated_people = []
    if pose_res.keypoints is not None:
        boxes = pose_res.boxes.xyxy.cpu().numpy().astype(int)
        kpts = pose_res.keypoints.xy.cpu().numpy()
        confs = pose_res.keypoints.conf.cpu().numpy()

        for i in range(len(boxes)):
            if is_sitting_posture(kpts[i], confs[i]):
                if confs[i][11] > 0.3 or confs[i][12] > 0.3:
                    hx = float((kpts[i][11][0] + kpts[i][12][0]) / 2.0)
                    hy = float((kpts[i][11][1] + kpts[i][12][1]) / 2.0)
                else:
                    hx = float((boxes[i][0] + boxes[i][2]) / 2.0)
                    hy = float(boxes[i][1] + (boxes[i][3] - boxes[i][1]) * 0.75)
                seated_people.append({"box": boxes[i], "hip": (hx, hy)})

    # 2. Chair detection (Class 56) - scale chair conf slightly relative to base threshold
    chair_conf = max(0.05, conf_threshold - 0.08)
    detect_res = detect_model.predict(frame, imgsz=imgsz, classes=[56], conf=chair_conf, device=DEVICE, verbose=False)[0]
    raw_chairs = []
    for box in detect_res.boxes:
        b = box.xyxy[0].cpu().numpy().astype(int)
        bw = b[2] - b[0]
        bh = b[3] - b[1]
        # Ignore wide benches or long tables misdetected as chairs
        if (bw / float(bh)) <= 1.6:
            raw_chairs.append(b)

    # 3. Associate chairs with sitting people
    occupied_count = 0
    vacant_count = 0

    for cbox in raw_chairs:
        is_occupied = any(
            box_contains_point(cbox, p["hip"]) or compute_iou(cbox, p["box"]) > 0.15
            for p in seated_people
        )
        if is_occupied:
            occupied_count += 1
        else:
            vacant_count += 1

    total_seats = occupied_count + vacant_count
    return total_seats, occupied_count, vacant_count


def run_camera_test(cam):
    source = cam["source"]
    conf_threshold = float(cam.get("conf_threshold", 0.25))
    is_live = source.startswith("rtsp://") or source.startswith("http://")
    if not is_live:
        source = os.path.join(BASE_DIR, source) if not os.path.isabs(source) else source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        cam["status"] = "Offline (Source unreachable)"
        cam["last_tested"] = datetime.now().strftime("%H:%M:%S")
        return

    ret, frame = False, None
    if is_live:
        # Flush stream buffer for live feeds to get the latest frame
        for _ in range(5):
            ret, frame = cap.read()
            if not ret:
                break
    else:
        ret, frame = cap.read()

    cap.release()

    if not ret or frame is None:
        cam["status"] = "No Video Signal"
        cam["last_tested"] = datetime.now().strftime("%H:%M:%S")
        return

    total, occ, vac = analyze_frame_dynamically(frame, conf_threshold)
    cam["total_seats"] = total
    cam["occupied"] = occ
    cam["vacant"] = vac
    cam["status"] = "Active"
    cam["last_tested"] = datetime.now().strftime("%H:%M:%S")


# ================= Background Polling Task =================
async def background_scheduler():
    last_exec = {}
    while True:
        now = time.time()
        updated = False
        for cam in CAMERAS:
            cid = cam["id"]
            interval = int(cam.get("interval_sec", 30))
            if now - last_exec.get(cid, 0) >= interval:
                last_exec[cid] = now
                await asyncio.to_thread(run_camera_test, cam)
                updated = True
        if updated:
            persist_store()
        await asyncio.sleep(1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_internal_store()
    task = asyncio.create_task(background_scheduler())
    yield
    task.cancel()

app = FastAPI(title="LibEdge Camera & Zone Manager", lifespan=lifespan)


# ================= REST Endpoints =================
class CameraModel(BaseModel):
    id: str
    name: str
    floor: str
    zone: str
    source: str
    interval_sec: int
    conf_threshold: float = 0.25


@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/api/cameras")
async def get_all_cameras():
    return JSONResponse(CAMERAS)


@app.post("/api/cameras")
async def save_camera(data: CameraModel):
    global CAMERAS
    for i, c in enumerate(CAMERAS):
        if c["id"] == data.id:
            CAMERAS[i]["name"] = data.name
            CAMERAS[i]["floor"] = data.floor
            CAMERAS[i]["zone"] = data.zone
            CAMERAS[i]["source"] = data.source
            CAMERAS[i]["interval_sec"] = data.interval_sec
            CAMERAS[i]["conf_threshold"] = data.conf_threshold
            persist_store()
            return {"status": "updated"}

    new_entry = {
        "id": data.id,
        "name": data.name,
        "floor": data.floor,
        "zone": data.zone,
        "source": data.source,
        "interval_sec": data.interval_sec,
        "conf_threshold": data.conf_threshold,
        "total_seats": 0,
        "occupied": 0,
        "vacant": 0,
        "last_tested": "Pending",
        "status": "Initialized"
    }
    CAMERAS.append(new_entry)
    persist_store()
    return {"status": "created"}


@app.delete("/api/cameras/{cam_id}")
async def delete_camera(cam_id: str):
    global CAMERAS
    CAMERAS = [c for c in CAMERAS if c["id"] != cam_id]
    cameras_collection.delete_one({"id": cam_id})
    persist_store()
    return {"status": "deleted"}


@app.post("/api/cameras/{cam_id}/test-now")
async def test_camera_now(cam_id: str):
    for c in CAMERAS:
        if c["id"] == cam_id:
            await asyncio.to_thread(run_camera_test, c)
            persist_store()
            return {
                "status": "success", 
                "total_seats": c["total_seats"], 
                "occupied": c["occupied"], 
                "vacant": c["vacant"]
            }
    return JSONResponse(status_code=404, content={"status": "not found"})
