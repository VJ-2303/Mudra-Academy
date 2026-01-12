"""
MUDRA DETECTION API SERVER
Flask API server that wraps the hybrid ML model for web integration.
Supports both single-hand (MediaPipe + RF) and double-hand (YOLO) detection.

KEY FIX: Added session-based state management with FSM to match Python webcam accuracy.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import base64
import numpy as np
from PIL import Image
from io import BytesIO
import mediapipe as mp
import threading
import time
import os
import sys
import google.generativeai as genai
from types import SimpleNamespace
from collections import defaultdict
import logging

# Configuration
API_PORT = 5000
DEBUG = True

# ============================================
# LOGGING CONFIGURATION
# ============================================
# Set to True to enable all logging, False to disable
ENABLE_LOGGING = False

if not ENABLE_LOGGING:
    # Disable Flask/Werkzeug request logging
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    # Disable other loggers
    logging.getLogger('PIL').setLevel(logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable TensorFlow logs
    
def log(message):
    """Print message only if logging is enabled."""
    if ENABLE_LOGGING:
        print(message)

sys.path.insert(0, os.path.dirname(__file__))

from hybrid_webcam import (
    model, model_classes,
    get_scale_ref,
    detect_mudra_hybrid,
    RULE_MUDRA_FUNCTIONS,
    ML_CONF_THRESHOLD,
    MudraFSM  # Import FSM for state management
)

# ============================================================
# Load Models
# ============================================================

# 0. Configure Gemini (Chatbot)
# ------------------------------------------------------------
# TODO: USER - REPLACE WITH YOUR ACTUAL API KEY
GEMINI_API_KEY = "AIzaSyA3eRUFLP0qAOZXuIltuUxhy8niX6-YODI"
genai.configure(api_key=GEMINI_API_KEY)

try:
    chat_model = genai.GenerativeModel('gemini-flash-latest')
    CHAT_LOADED = True
    print("✓ Gemini Chat model configured")
except Exception as e:
    print(f"✗ Failed to configure Gemini: {e}")
    CHAT_LOADED = False

# 1. Load Random Forest Model (Single Hand)
# ------------------------------------------------------------
# The RF model is loaded within hybrid_webcam.py

# ============================================
# YOLO Model for Double-Hand Detection
# ============================================
try:
    from ultralytics import YOLO
    YOLO_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'kkpv.pt')
    if os.path.exists(YOLO_MODEL_PATH):
        yolo_model = YOLO(YOLO_MODEL_PATH)
        YOLO_LOADED = True
        log(f"YOLO model loaded from {YOLO_MODEL_PATH}")
    else:
        yolo_model = None
        YOLO_LOADED = False
        log(f"YOLO model not found at {YOLO_MODEL_PATH}")
except ImportError:
    yolo_model = None
    YOLO_LOADED = False
    log("ultralytics not installed - double-hand detection disabled")

# Double-hand mudra class names (matching the trained model exactly)
DOUBLE_HAND_CLASSES = {
    0: "Anjali",
    1: "Naagabandha",
    2: "Chakra",
    3: "Karkota",
    4: "Katariswastika",
    5: "Pasha",
    6: "Shanka",
    7: "Shivalinga"
}

DOUBLE_HAND_MEANINGS = {
    "Anjali": "Salutation - Palms joined in prayer position",
    "Naagabandha": "Serpent bond - Intertwined snakes",
    "Chakra": "Disc/Wheel - Vishnu's divine weapon",
    "Karkota": "Crab - Interlocked fingers showing strength",
    "Katariswastika": "Crossed scissors - Both hands in Kartarimukha crossed",
    "Pasha": "Noose - Rope or binding",
    "Shanka": "Conch shell - Sacred symbol",
    "Shivalinga": "Shiva's symbol - Sacred emblem of Lord Shiva"
}
app = Flask(__name__)
CORS(app)

mp_hands = mp.solutions.hands

# Note: We'll create per-session MediaPipe hands instances for proper tracking

SUPPORTED_MUDRAS = sorted(set(list(RULE_MUDRA_FUNCTIONS.keys()) + list(model_classes)))

# ============================================
# SESSION STATE MANAGEMENT
# ============================================
# Store per-session state for FSM, previous landmarks, AND MediaPipe hands instance
# This fixes the accuracy issue where ML model couldn't work without prev_landmarks

class SessionState:
    """Maintains state for a detection session (per client)."""
    def __init__(self):
        self.fsm = MudraFSM()
        # SPEED FIX: Reduce FSM thresholds for faster web response
        self.fsm.ENTER_THRESHOLD = 2  # Reduced from 3 to 2 frames for faster ML confirmation
        self.fsm.EXIT_THRESHOLD = 2
        self.fsm.MAX_MISMATCH = 1
        
        self.prev_landmarks = None
        self.last_access = time.time()
        # Per-session MediaPipe hands for proper tracking between frames
        self.hands = mp_hands.Hands(
            static_image_mode=False,  # Video mode for better tracking
            max_num_hands=1,
            min_detection_confidence=0.6,  # SPEED FIX: Lowered from 0.7 for faster detection
            min_tracking_confidence=0.6    # SPEED FIX: Lowered from 0.7 for faster tracking
        )
    
    def update_landmarks(self, landmarks):
        """Store current landmarks as previous for next frame."""
        self.prev_landmarks = [SimpleNamespace(x=lm.x, y=lm.y, z=lm.z) for lm in landmarks]
        self.last_access = time.time()
    
    def close(self):
        """Clean up MediaPipe resources."""
        if self.hands:
            self.hands.close()

# Session storage (simple in-memory, keyed by session_id)
sessions = defaultdict(SessionState)
SESSION_TIMEOUT = 60  # seconds - clean up old sessions

def get_session(session_id):
    """Get or create a session state."""
    if session_id is None:
        session_id = "default"
    
    # Clean up old sessions periodically
    current_time = time.time()
    expired = [sid for sid, state in sessions.items() 
               if current_time - state.last_access > SESSION_TIMEOUT]
    for sid in expired:
        sessions[sid].close()  # Clean up MediaPipe resources
        del sessions[sid]
    
    return sessions[session_id]


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
    """
    Single-hand mudra detection with FSM state management.
    
    This endpoint now maintains session state to:
    1. Track previous landmarks for hand stability detection
    2. Use FSM for debouncing (requires 3 consecutive frames for ML mudras)
    3. Match the accuracy of the standalone Python webcam
    
    Optional request parameters:
    - session_id: Unique ID to maintain state across requests (default: "default")
    - include_landmarks: If true, include landmark coordinates in response
    """
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({"success": False, "error": "No image provided"}), 400
        
        # Get or create session state
        session_id = data.get('session_id', 'default')
        session = get_session(session_id)
        
        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Resize to consistent size (matching Python webcam)
        frame = cv2.resize(frame, (640, 480))
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Use session's MediaPipe hands for proper tracking between frames
        results = session.hands.process(frame_rgb)
        
        if not results.multi_hand_landmarks:
            # No hand - update FSM state
            display_text, _, _ = session.fsm.update(
                hand_present=False,
                candidate_name=None,
                candidate_conf=0.0,
                candidate_method=None
            )
            session.prev_landmarks = None
            
            return jsonify({
                "success": True,
                "hand_detected": False,
                "mudra": "No hand detected",
                "confidence": 0.0,
                "method": "NONE",
                "fsm_state": session.fsm.state
            }), 200
        
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = hand_landmarks.landmark
        
        handedness = results.multi_handedness[0] if results.multi_handedness else None
        handedness_label = handedness.classification[0].label if handedness else "Right"
        
        # Run hybrid detection with previous landmarks for stability check
        mudra_name, confidence, method = detect_mudra_hybrid(
            landmarks, 
            handedness_label, 
            prev_landmarks=session.prev_landmarks  # FIX: Now passing previous landmarks
        )
        
        # Normalize candidate for FSM
        if mudra_name in ["Unknown", "Stabilizing..."]:
            candidate_name = None
            candidate_conf = 0.0
            candidate_method = None
        else:
            candidate_name = mudra_name
            candidate_conf = confidence
            candidate_method = method
        
        # Update FSM with detection result
        display_text, disp_conf, disp_method = session.fsm.update(
            hand_present=True,
            candidate_name=candidate_name,
            candidate_conf=candidate_conf,
            candidate_method=candidate_method
        )
        
        # Store current landmarks for next frame's stability check
        session.update_landmarks(landmarks)
        
        # Determine final output
        if display_text in ["Show mudra...", "Detecting...", "Stabilizing..."]:
            final_mudra = "Unknown"
            final_conf = 0.0
            final_method = "NONE"
        else:
            final_mudra = display_text
            final_conf = disp_conf
            final_method = disp_method if disp_method else "NONE"
        
        response_data = {
            "success": True,
            "hand_detected": True,
            "mudra": final_mudra,
            "confidence": float(final_conf),
            "method": final_method,
            "fsm_state": session.fsm.state,
            "raw_detection": mudra_name  # For debugging
        }
        
        if data.get('include_landmarks', False):
            response_data['landmarks'] = [[lm.x, lm.y, lm.z] for lm in landmarks]
        
        return jsonify(response_data), 200
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# Endpoint to reset a session's FSM state
@app.route('/reset-session', methods=['POST'])
def reset_session():
    """Reset FSM state for a session (useful when user clicks Reset)."""
    try:
        data = request.get_json() or {}
        session_id = data.get('session_id', 'default')
        
        if session_id in sessions:
            sessions[session_id].close()  # Clean up old MediaPipe instance
            sessions[session_id] = SessionState()  # Create fresh session
        
        return jsonify({
            "success": True, 
            "message": "Session reset successfully",
            "fsm_state": "S0_NO_HAND"
        }), 200

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/chat', methods=['POST'])
def chat():
    """
    Chat endpoint using Gemini.
    Expects JSON: { "message": "User question..." }
    """
    if not CHAT_LOADED:
        return jsonify({"success": False, "error": "Chat system not authorized (Key missing?)"}), 503

    try:
        data = request.get_json()
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({"success": False, "error": "Empty message"}), 400

        # System Prompt Injection
        system_prompt = (
            "You are 'Natya Guru', a wise and encouraging Bharatanatyam teacher. "
            "You help students understand Mudras (hand gestures), their mythology, uses (Viniyoga), and spiritual significance. "
            "Keep answers concise, educational, and use a warm, mentorship tone. "
            "If asked about app navigation, guide them gently. "
            "Dont give table formatted responses"
            f"User asks: {user_message}"
        )

        response = chat_model.generate_content(system_prompt)
        bot_reply = response.text

        return jsonify({
            "success": True,
            "reply": bot_reply
        })

    except Exception as e:
        print(f"Chat Error: {e}")
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
    log("=" * 60)
    log("MUDRA DETECTION API SERVER")
    log("=" * 60)
    log(f"Single-hand mudras: {len(SUPPORTED_MUDRAS)}")
    log(f"  - Rule-based: {len(RULE_MUDRA_FUNCTIONS)}")
    log(f"  - ML-based: {len(model_classes)}")
    log(f"Double-hand mudras: {len(DOUBLE_HAND_CLASSES) if YOLO_LOADED else 'Not loaded'}")
    log(f"YOLO model: {'Loaded' if YOLO_LOADED else 'Not available'}")
    log("-" * 60)
    log(f"Logging: {'ENABLED' if ENABLE_LOGGING else 'DISABLED'}")
    log(f"Server starting on http://localhost:5000")
    log("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
