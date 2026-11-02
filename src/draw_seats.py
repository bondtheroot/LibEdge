
import cv2
import json
import os

IMAGE_PATH = "outputs/seat_layout.jpg"
OUTPUT_PATH = "dataset/annotations/seats.json"

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        f"Cannot open {IMAGE_PATH}. Extract the image first."
    )

display = image.copy()
points = []
seats = []

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)


def redraw():
    global display
    display = image.copy()

    # Draw previously saved seats
    for seat in seats:
        poly = seat["polygon"]
        pts = __import__("numpy").array(poly, dtype="int32")
        cv2.polylines(display, [pts], True, (0, 255, 0), 2)

        x, y = poly[0]
        cv2.putText(
            display, seat["id"], (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0, 255, 0), 2
        )

    # Draw the polygon currently being created
    for p in points:
        cv2.circle(display, p, 5, (0, 0, 255), -1)

    if len(points) > 1:
        for i in range(1, len(points)):
            cv2.line(
                display, points[i - 1], points[i],
                (0, 0, 255), 2
            )

    if len(points) > 2:
        cv2.line(display, points[-1], points[0], (0, 0, 255), 1)

    instructions = "Click corners | S: save seat | U: undo | R: reset | Q: finish"
    cv2.putText(
        display, instructions, (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
        (255, 255, 0), 2
    )


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        redraw()


window = "LibEdge Seat Polygon Tool"
cv2.namedWindow(window, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(window, mouse_callback)

print("Click 4 or more points around each chair.")
print("Press S to save a seat, U to undo, R to reset, Q to finish.")

while True:
    redraw()
    cv2.imshow(window, display)
    key = cv2.waitKey(20) & 0xFF

    if key in (ord("s"), ord("S")):
        if len(points) < 3:
            print("A polygon needs at least 3 points.")
            continue

        seat_id = f"S{len(seats) + 1:02d}"
        seats.append({
            "id": seat_id,
            "zone": "A",
            "polygon": points.copy(),
            "adjacent_to": []
        })

        print(f"Saved {seat_id}: {points}")
        points.clear()

    elif key in (ord("u"), ord("U")):
        if points:
            points.pop()

    elif key in (ord("r"), ord("R")):
        points.clear()

    elif key in (ord("q"), ord("Q"), 27):
        break

cv2.destroyAllWindows()

with open(OUTPUT_PATH, "w") as f:
    json.dump({"seats": seats}, f, indent=2)

print(f"\nSaved {len(seats)} seats to {OUTPUT_PATH}")

