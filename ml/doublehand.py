import cv2
from ultralytics import YOLO
import numpy as np
import time
import os

# -----------------------------
# Load YOLOv8 Detection Model
# -----------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'kkpv.pt')
model = YOLO(MODEL_PATH)

# Class names from your dataset
class_names = ['Anjali', 'Naagabandha', 'chakra', 'karkota', 'katariswastika', 'pasha', 'shanka', 'shivalinga']

# Colors for each class (BGR format)
colors = [
    (147, 112, 219),  # Anjali - purple
    (107, 107, 255),  # Naagabandha - red
    (237, 149, 100),  # chakra - cornflower blue
    (0, 215, 255),    # karkota - gold
    (78, 205, 196),   # katariswastika - teal
    (69, 183, 209),   # pasha - blue
    (200, 216, 152),  # shanka - light green
    (247, 220, 111)   # shivalinga - yellow
]

# -----------------------------
# Open Webcam
# -----------------------------
cap = cv2.VideoCapture(0)

# Set camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

print("Starting webcam detection...")
print("Press 'q' to quit")
print("Press 's' to save current frame")

frame_count = 0
prev_time = time.time()

# -----------------------------
# Webcam Detection Loop
# -----------------------------
while True:
    # Read frame from webcam
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    # Run YOLOv8 inference
    results = model(frame, conf=0.5, iou=0.4, verbose=False)
    
    # Get the first result
    result = results[0]
    
    # Create a copy of the frame for drawing
    annotated_frame = frame.copy()
    
    # Draw custom bounding boxes with your colors
    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Get class and confidence
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            # Get color for this class
            color = colors[cls] if cls < len(colors) else (0, 255, 0)
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label
            label = f'{class_names[cls]}: {conf:.2f}'
            
            # Get label size for background
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Draw label background
            cv2.rectangle(
                annotated_frame,
                (x1, y1 - label_height - 10),
                (x1 + label_width, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated_frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
    
    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    
    # Display detection count
    detection_count = len(result.boxes) if result.boxes is not None else 0
    
    # Add info overlay with semi-transparent background
    overlay = annotated_frame.copy()
    cv2.rectangle(overlay, (10, 10), (300, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, annotated_frame, 0.6, 0, annotated_frame)
    
    # Add text information
    cv2.putText(
        annotated_frame,
        f'Detections: {detection_count}',
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )
    
    cv2.putText(
        annotated_frame,
        f'FPS: {fps:.1f}',
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )
    
    # Display the frame
    cv2.imshow('Mudra Detection - Press Q to quit, S to save', annotated_frame)
    
    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q') or key == ord('Q'):
        print("Quitting...")
        break
    elif key == ord('s') or key == ord('S'):
        # Save current frame
        filename = f'detection_{frame_count}.jpg'
        cv2.imwrite(filename, annotated_frame)
        print(f"Saved frame as {filename}")
    
    frame_count += 1

# -----------------------------
# Cleanup
# -----------------------------
cap.release()
cv2.destroyAllWindows()
print("Webcam detection stopped.")