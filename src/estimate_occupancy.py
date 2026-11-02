
import cv2
import json
import numpy as np
from ultralytics import YOLO

VIDEO = "dataset/raw/library_01.mp4"
SEATS_FILE = "dataset/annotations/seats.json"
OUTPUT = "outputs/occupancy_video.mp4"

MODEL_PATH = "yolo11n.pt"
CONFIDENCE = 0.35
IMAGE_SIZE = 640

# Initial heuristic; tune against manually labelled frames.
OVERLAP_THRESHOLD = 0.20

model = YOLO(MODEL_PATH)

with open(SEATS_FILE, "r") as f:
    seats = json.load(f)["seats"]

if not seats:
    raise ValueError("No seats found in seats.json")

cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 0:
    fps = 25.0

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if width <= 0 or height <= 0:
    cap.release()
    raise RuntimeError("Could not read video dimensions")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT, fourcc, fps, (width, height))

if not writer.isOpened():
    cap.release()
    raise RuntimeError(f"Could not create output video: {OUTPUT}")


def make_seat_mask(polygon):
    """Create a binary mask for one seat polygon."""
    mask = np.zeros((height, width), dtype=np.uint8)
    points = np.asarray(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [points], 1)
    return mask


# Prepare seat masks once, rather than rebuilding them per frame.
seat_data = []

for seat in seats:
    polygon = seat.get("polygon", [])

    if len(polygon) < 3:
        print(f"Skipping invalid polygon: {seat.get('id', 'unknown')}")
        continue

    mask = make_seat_mask(polygon)
    area = int(mask.sum())

    if area == 0:
        print(f"Skipping zero-area seat: {seat.get('id', 'unknown')}")
        continue

    seat_data.append({
        "id": seat["id"],
        "polygon": np.asarray(polygon, dtype=np.int32),
        "mask": mask,
        "area": area,
    })

if not seat_data:
    cap.release()
    writer.release()
    raise ValueError("No valid seat polygons found")


frame_number = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    result = model.predict(
        frame,
        classes=[0],       # Person class in the COCO model
        conf=CONFIDENCE,
        imgsz=IMAGE_SIZE,
        verbose=False,
    )[0]

    # Store person bounding boxes.
    person_boxes = []

    if result.boxes is not None:
        person_boxes = (
            result.boxes.xyxy.cpu().numpy().astype(int).tolist()
        )

    occupied_count = 0

    for seat in seat_data:
        seat_mask = seat["mask"]
        seat_area = seat["area"]
        occupied = False
        best_overlap = 0.0

        for box in person_boxes:
            x1, y1, x2, y2 = box

            # Clip box coordinates to the video frame.
            x1 = max(0, min(width, x1))
            y1 = max(0, min(height, y1))
            x2 = max(0, min(width, x2))
            y2 = max(0, min(height, y2))

            if x2 <= x1 or y2 <= y1:
                continue

            # Count seat pixels inside this person's box.
            overlap_pixels = int(
                seat_mask[y1:y2, x1:x2].sum()
            )

            # Fraction of the seat polygon covered by the box.
            overlap_ratio = overlap_pixels / seat_area
            best_overlap = max(best_overlap, overlap_ratio)

            if overlap_ratio >= OVERLAP_THRESHOLD:
                occupied = True

        if occupied:
            occupied_count += 1
            color = (0, 0, 255)      # Red = estimated occupied
            status = "OCCUPIED"
        else:
            color = (0, 200, 0)      # Green = estimated empty
            status = "EMPTY"

        cv2.polylines(
            frame,
            [seat["polygon"]],
            True,
            color,
            2,
        )

        x, y = seat["polygon"][0]
        label = f'{seat["id"]}: {status}'

        cv2.putText(
            frame,
            label,
            (int(x), max(20, int(y) - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    # Draw person detections for debugging.
    for x1, y1, x2, y2 in person_boxes:
        cv2.rectangle(
            frame, (x1, y1), (x2, y2),
            (255, 180, 0), 1
        )

    summary = (
        f"Estimated occupied: {occupied_count}/{len(seat_data)}"
    )

    cv2.putText(
        frame,
        summary,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    writer.write(frame)
    frame_number += 1

    if frame_number % 100 == 0:
        print(f"Processed {frame_number} frames")

cap.release()
writer.release()

print(f"Finished processing {frame_number} frames.")
print(f"Valid seats: {len(seat_data)}")
print(f"Output saved to: {OUTPUT}")

