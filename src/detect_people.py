import cv2
from ultralytics import YOLO

VIDEO = "dataset/raw/library_01.mp4"
OUTPUT = "outputs/detected_video.mp4"

# Small pretrained model for the first experiment
model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps <= 0:
    fps = 25.0

writer = cv2.VideoWriter(
    OUTPUT,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)

frame_number = 0

while True:
    ok, frame = cap.read()

    if not ok:
        break

    result = model.predict(
        frame,
        classes=[0],  # Person class
        conf=0.35,
        imgsz=640,
        verbose=False,
    )[0]

    annotated = result.plot()
    writer.write(annotated)

    frame_number += 1

    if frame_number % 100 == 0:
        print("Processed frames:", frame_number)

cap.release()
writer.release()

print("Finished. Output:", OUTPUT)
