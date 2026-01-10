"""
MUDRA DETECTION API SERVER
Flask API server that wraps the hybrid ML model for web integration.
Supports both single-hand (MediaPipe + RF) and double-hand (YOLO) detection.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import mediapipe as mp
import base64
from io import BytesIO
from PIL import Image
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from hybrid_webcam import (
    model, model_classes,
    get_scale_ref,
    detect_mudra_hybrid,
    RULE_MUDRA_FUNCTIONS,
    ML_CONF_THRESHOLD
)

# ============================================
# YOLO Model for Double-Hand Detection
# ============================================
try:
    from ultralytics import YOLO
    YOLO_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best.pt')
    if os.path.exists(YOLO_MODEL_PATH):
        yolo_model = YOLO(YOLO_MODEL_PATH)
        YOLO_LOADED = True
        print(f"YOLO model loaded from {YOLO_MODEL_PATH}")
    else:
        yolo_model = None
        YOLO_LOADED = False
        print(f"YOLO model not found at {YOLO_MODEL_PATH}")
except ImportError:
    yolo_model = None
    YOLO_LOADED = False
    print("ultralytics not installed - double-hand detection disabled")

# Double-hand mudra class names (matching the trained model exactly)
DOUBLE_HAND_CLASSES = {
    0: "Chakra",
    1: "Karkota",
    2: "Katariswastika",
    3: "Nagabandha",
    4: "Pasha",
    5: "Shanka"
}

DOUBLE_HAND_MEANINGS = {
    "Chakra": "Disc/Wheel - Vishnu's divine weapon",
    "Karkota": "Crab - Interlocked fingers showing strength",
    "Katariswastika": "Crossed scissors - Both hands in Kartarimukha crossed",
    "Nagabandha": "Serpent bond - Intertwined snakes",
    "Pasha": "Noose - Rope or binding",
    "Shanka": "Conch shell - Sacred symbol"
}

app = Flask(__name__)
CORS(app)

print(f"ML Model loaded with {len(model_classes)} classes")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

SUPPORTED_MUDRAS = sorted(set(list(RULE_MUDRA_FUNCTIONS.keys()) + list(model_classes)))

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": True,
        "mudra_count": len(SUPPORTED_MUDRAS),
        "rule_based_mudras": len(RULE_MUDRA_FUNCTIONS),
        "ml_mudras": len(model_classes),
        "double_hand_model": YOLO_LOADED,
        "double_hand_mudras": len(DOUBLE_HAND_CLASSES) if YOLO_LOADED else 0
    }), 200

@app.route('/mudras', methods=['GET'])
def get_mudras():
    return jsonify({
        "mudras": SUPPORTED_MUDRAS,
        "count": len(SUPPORTED_MUDRAS),
        "rule_based": list(RULE_MUDRA_FUNCTIONS.keys()),
        "ml_based": model_classes.tolist()
    }), 200

@app.route('/detect', methods=['POST'])
def detect():
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({"success": False, "error": "No image provided"}), 400
        
        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        if not results.multi_hand_landmarks:
            return jsonify({
                "success": True,
                "hand_detected": False,
                "mudra": "No hand detected",
                "confidence": 0.0,
                "method": "NONE"
            }), 200
        
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = hand_landmarks.landmark
        
        handedness = results.multi_handedness[0] if results.multi_handedness else None
        handedness_label = handedness.classification[0].label if handedness else "Right"
        
        mudra_name, confidence, method = detect_mudra_hybrid(
            landmarks, 
            handedness_label, 
            prev_landmarks=None
        )
        
        if mudra_name == "Stabilizing...":
            mudra_name = "Unknown"
            confidence = 0.0
            method = "NONE"
        
        if method is None:
            method = "NONE"
        
        response_data = {
            "success": True,
            "hand_detected": True,
            "mudra": mudra_name,
            "confidence": float(confidence),
            "method": method
        }
        
        if data.get('include_landmarks', False):
            response_data['landmarks'] = [[lm.x, lm.y, lm.z] for lm in landmarks]
        
        return jsonify(response_data), 200
        
    except Exception as e:
        import traceback
        print(f"Error: {traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500


# ============================================
# DOUBLE-HAND DETECTION ENDPOINT (YOLO)
# ============================================
@app.route('/detect-double', methods=['POST'])
def detect_double():
    """Detect double-hand mudras using YOLO model"""
    try:
        if not YOLO_LOADED or yolo_model is None:
            return jsonify({
                "success": False,
                "error": "Double-hand model not loaded"
            }), 503
        
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({"success": False, "error": "No image provided"}), 400
        
        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Run YOLO inference
        results = yolo_model(frame, conf=0.5, verbose=False)
        
        mudra_name = "Unknown"
        confidence = 0.0
        bbox = None
        
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            
            # Get the detection with highest confidence
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            
            if len(scores) > 0:
                best_idx = np.argmax(scores)
                cls_id = int(classes[best_idx])
                mudra_name = DOUBLE_HAND_CLASSES.get(cls_id, "Unknown")
                confidence = float(scores[best_idx])
                box = boxes[best_idx]
                bbox = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
        
        response_data = {
            "success": True,
            "mudra": mudra_name,
            "confidence": confidence,
            "meaning": DOUBLE_HAND_MEANINGS.get(mudra_name, ""),
            "method": "YOLO"
        }
        
        if bbox:
            response_data["bbox"] = bbox
        
        return jsonify(response_data), 200
        
    except Exception as e:
        import traceback
        print(f"Double-hand detection error: {traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/double-mudras', methods=['GET'])
def get_double_mudras():
    """Get list of supported double-hand mudras"""
    return jsonify({
        "mudras": list(DOUBLE_HAND_CLASSES.values()),
        "count": len(DOUBLE_HAND_CLASSES),
        "meanings": DOUBLE_HAND_MEANINGS,
        "model_loaded": YOLO_LOADED
    }), 200


if __name__ == '__main__':
    print("=" * 60)
    print("MUDRA DETECTION API SERVER")
    print("=" * 60)
    print(f"Single-hand mudras: {len(SUPPORTED_MUDRAS)}")
    print(f"  - Rule-based: {len(RULE_MUDRA_FUNCTIONS)}")
    print(f"  - ML-based: {len(model_classes)}")
    print(f"Double-hand mudras: {len(DOUBLE_HAND_CLASSES) if YOLO_LOADED else 'Not loaded'}")
    print(f"YOLO model: {'Loaded' if YOLO_LOADED else 'Not available'}")
    print("-" * 60)
    print(f"Server starting on http://localhost:5000")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
