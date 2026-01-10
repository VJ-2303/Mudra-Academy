import cv2
from ultralytics import YOLO
import os

# -----------------------------
# Load YOLOv8 Detection Model
# -----------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best.pt')
model = YOLO(MODEL_PATH)

# Class names (matching trained model exactly)
class_names = {
    0: "Chakra",
    1: "Karkota",
    2: "Katariswastika",
    3: "Nagabandha",
    4: "Pasha",
    5: "Shanka"
}

# -----------------------------
# Open Webcam
# -----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

print("✅ Webcam started. Press 'q' to quit.")

# -----------------------------
# Webcam Detection Loop
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8 inference
    results = model(frame, conf=0.5)

    # Draw detections
    for r in results:
        if r.boxes is None:
            continue

        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()

        for box, score, cls_id in zip(boxes, scores, classes):
            cls_id = int(cls_id)
            label = class_names.get(cls_id, "unknown")
            confidence = f"{score:.2f}"

            x1, y1, x2, y2 = map(int, box)

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Label text
            cv2.putText(
                frame,
                f"{label} {confidence}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    cv2.imshow("Bharatanatyam Mudra Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# -----------------------------
# Cleanup
# -----------------------------
cap.release()
cv2.destroyAllWindows()