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
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

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
        # Default seed entry with explicit thresholds matching estimate_occupancy.py
        CAMERAS = [
            {
                "id": "cam_01",
                "name": "Main Reading Hall",
                "floor": "Floor 1",
                "zone": "Quiet Zone A",
                "source": "dataset/raw/library.mp4",
                "interval_sec": 30,
                "person_conf": 0.25,
                "chair_conf": 0.17,
                "total_seats": 0,
                "occupied": 0,
                "vacant": 0,
                "seats": [],
                "output_video": "",
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


def analyze_frame_dynamically(frame, person_conf=0.25, chair_conf=0.17):
    """Detects chairs and sitting people dynamically, clustering chairs into rows and structured IDs."""
    h, w = frame.shape[:2]
    imgsz = min(max(w, h), 960)

    # 1. Pose estimation for people
    pose_res = pose_model.predict(frame, imgsz=imgsz, conf=person_conf, device=DEVICE, verbose=False)[0]
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

    # 2. Chair detection (Class 56)
    detect_res = detect_model.predict(frame, imgsz=imgsz, classes=[56], conf=chair_conf, device=DEVICE, verbose=False)[0]
    raw_chairs = []
    for box in detect_res.boxes:
        b = box.xyxy[0].cpu().numpy().astype(int)
        cls_id = int(box.cls[0].cpu().numpy())
        conf = float(box.conf[0].cpu().numpy())
        bw = b[2] - b[0]
        bh = b[3] - b[1]
        if cls_id == 56 and conf >= chair_conf and (bw / float(bh)) <= 1.6:
            raw_chairs.append(b)

    # 3. Spatial Row Clustering & Column Sorting
    raw_chairs = sorted(raw_chairs, key=lambda c: c[1])
    rows = []
    current_row = []
    row_threshold = 45  # Pixel tolerance to group chairs into the same horizontal row

    for c in raw_chairs:
        if not current_row:
            current_row.append(c)
        else:
            avg_y = sum(item[1] for item in current_row) / len(current_row)
            if abs(c[1] - avg_y) <= row_threshold:
                current_row.append(c)
            else:
                rows.append(current_row)
                current_row = [c]
    if current_row:
        rows.append(current_row)

    structured_seats = []
    occupied_count = 0
    vacant_count = 0

    for r_idx, row in enumerate(rows):
        row_sorted = sorted(row, key=lambda c: c[0])
        for s_idx, cbox in enumerate(row_sorted):
            seat_id = f"R{r_idx+1}S{s_idx+1}"
            is_occupied = any(
                box_contains_point(cbox, p["hip"]) or compute_iou(cbox, p["box"]) > 0.15
                for p in seated_people
            )
            if is_occupied:
                occupied_count += 1
                status = "Occupied"
            else:
                vacant_count += 1
                status = "Vacant"

            structured_seats.append({
                "seat_id": seat_id,
                "row": r_idx + 1,
                "seat": s_idx + 1,
                "box": cbox.tolist(),
                "status": status
            })

    total_seats = occupied_count + vacant_count
    return total_seats, occupied_count, vacant_count, structured_seats


def render_display(frame, structured_seats):
    overlay = frame.copy()
    occ_count = sum(1 for s in structured_seats if s["status"] == "Occupied")
    vac_count = sum(1 for s in structured_seats if s["status"] == "Vacant")

    for s in structured_seats:
        b = s["box"]
        is_occ = s["status"] == "Occupied"
        color = (52, 152, 219) if is_occ else (46, 204, 113)
        
        if not is_occ:
            cv2.rectangle(overlay, (b[0], b[1]), (b[2], b[3]), color, -1)
        cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), color, 2)
        cv2.putText(frame, f"{s['seat_id']}", (b[0], max(18, b[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

    banner = f"Vacant: {vac_count}  |  Occupied: {occ_count}"
    cv2.rectangle(frame, (8, 6), (280, 36), (20, 20, 20), -1)
    cv2.putText(frame, banner, (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

    return cv2.addWeighted(overlay, 0.20, frame, 0.80, 0)


def run_camera_test(cam, duration_sec=0):
    source = cam["source"]
    person_conf = float(cam.get("person_conf", 0.25))
    chair_conf = float(cam.get("chair_conf", 0.17))
    is_live = source.startswith("rtsp://") or source.startswith("http://")
    if not is_live:
        source = os.path.join(BASE_DIR, source) if not os.path.isabs(source) else source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        cam["status"] = "Offline (Source unreachable)"
        cam["last_tested"] = datetime.now().strftime("%H:%M:%S")
        return

    if duration_sec > 0:
        # 5-second accuracy test recording window (.webm format using VP80 for browser compatibility)
        out_path = os.path.join(OUTPUTS_DIR, f"{cam['id']}_test.webm")
        native_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        native_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 25.0

        fourcc = cv2.VideoWriter_fourcc(*"VP80")
        out = cv2.VideoWriter(out_path, fourcc, fps, (native_w, native_h))

        start_time = time.time()
        totals, occs, vacs, last_seats = [], [], [], []
        frame_idx = 0
        FRAME_STRIDE = 2
        cached_seats = []
        cached_tot, cached_occ, cached_vac = 0, 0, 0
        
        while time.time() - start_time < duration_sec:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break
            
            frame_idx += 1
            if frame_idx % FRAME_STRIDE == 0 or frame_idx == 1:
                cached_tot, cached_occ, cached_vac, cached_seats = analyze_frame_dynamically(frame, person_conf, chair_conf)

            totals.append(cached_tot)
            occs.append(cached_occ)
            vacs.append(cached_vac)
            last_seats = cached_seats
            
            rendered = render_display(frame, cached_seats)
            out.write(rendered)

        cap.release()
        out.release()

        if not totals:
            cam["status"] = "No Video Signal"
            cam["last_tested"] = datetime.now().strftime("%H:%M:%S")
            return

        cam["total_seats"] = int(round(sum(totals) / len(totals)))
        cam["occupied"] = int(round(sum(occs) / len(occs)))
        cam["vacant"] = int(round(sum(vacs) / len(vacs)))
        cam["seats"] = last_seats
        cam["output_video"] = f"/outputs/{cam['id']}_test.webm"
        cam["status"] = "Active (5s Accuracy Tested)"
    else:
        ret, frame = False, None
        if is_live:
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

        total, occ, vac, seats = analyze_frame_dynamically(frame, person_conf, chair_conf)
        cam["total_seats"] = total
        cam["occupied"] = occ
        cam["vacant"] = vac
        cam["seats"] = seats
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
                await asyncio.to_thread(run_camera_test, cam, 0)
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

# Mount outputs directory so rendered test videos can be viewed/downloaded
from fastapi.staticfiles import StaticFiles
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")


# ================= REST Endpoints =================
class CameraModel(BaseModel):
    id: str
    name: str
    floor: str
    zone: str
    source: str
    interval_sec: int
    person_conf: float = 0.25
    chair_conf: float = 0.17


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
            CAMERAS[i]["person_conf"] = data.person_conf
            CAMERAS[i]["chair_conf"] = data.chair_conf
            persist_store()
            return {"status": "updated"}

    new_entry = {
        "id": data.id,
        "name": data.name,
        "floor": data.floor,
        "zone": data.zone,
        "source": data.source,
        "interval_sec": data.interval_sec,
        "person_conf": data.person_conf,
        "chair_conf": data.chair_conf,
        "total_seats": 0,
        "occupied": 0,
        "vacant": 0,
        "seats": [],
        "output_video": "",
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
            # Trigger 5-second accuracy test recording video
            await asyncio.to_thread(run_camera_test, c, 5.0)
            persist_store()
            return {
                "status": "success", 
                "total_seats": c["total_seats"], 
                "occupied": c["occupied"], 
                "vacant": c["vacant"],
                "seats": c["seats"],
                "output_video": c.get("output_video", "")
            }
    return JSONResponse(status_code=404, content={"status": "not found"})
