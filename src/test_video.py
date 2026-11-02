import cv2

path = "dataset/raw/library_01.mp4"
cap = cv2.VideoCapture(path)

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {path}")

fps = cap.get(cv2.CAP_PROP_FPS)
count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Resolution:", width, "x", height)
print("FPS:", fps)
print("Frame count:", count)
print("Duration:", count / fps if fps > 0 else "Unknown", "seconds")

cap.release()
