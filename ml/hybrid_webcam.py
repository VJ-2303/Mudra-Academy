"""
================================================================================
HYBRID HAND MUDRA DETECTION SYSTEM (FSM)
================================================================================
Combines rule-based logic (strict geometric checks) with ML prediction (Random Forest)
in a single decision pipeline with Finite State Machine for flicker-free output:

PIPELINE:
    Webcam frame
       ↓
    MediaPipe Hands (detect landmarks)
       ↓
    Rule-based mudra checks (STRICT geometric rules)
       ↓
    IF rule confidently matches (score = 1.0):
           Accept rule result (immediate confirmation)
    ELSE:
           IF hand is steady:
                Extract ML features
                Run ML model
                IF ML confidence ≥ threshold:
                    Accept ML result (with debounce)
                ELSE:
                    "Unknown"
           ELSE:
                "Stabilizing..." (hand moving)
       ↓
    FSM State Machine (hysteresis for flicker-free transitions)
       ↓
    Display final result

FSM STATES:
- NO_HAND: No hand detected
- HAND_DETECTED: Hand present, awaiting mudra
- ENTERING_MUDRA: ML candidate being confirmed (debounce)
- CONFIRMED_MUDRA: Mudra locked and displaying
- EXITING_MUDRA: Mudra changing, awaiting confirmation

RESPONSIBILITIES:
- Rules: Detect mudras with clear geometric patterns (immediate confirmation)
- ML: Detect ambiguous mudras (requires 3 consecutive frames)
- FSM: Prevents flickering with entry/exit hysteresis

Author: Hybrid System
Date: December 30, 2025
================================================================================
"""

import cv2
import numpy as np
import mediapipe as mp
import joblib
import math
import os
from types import SimpleNamespace

# ============================================================
# FINITE STATE MACHINE FOR MUDRA DETECTION
# ============================================================

class MudraFSM:
    """
    Finite State Machine for robust mudra detection with hysteresis.
    
    States:
        - NO_HAND: No hand detected in frame
        - HAND_DETECTED: Hand present but no mudra yet
        - ENTERING_MUDRA: Accumulating evidence for a mudra (debouncing)
        - CONFIRMED_MUDRA: Mudra locked in and displaying
        - EXITING_MUDRA: Mudra changing, waiting for confirmation
    
    This eliminates flickering by requiring multiple consecutive frames
    before confirming entry/exit of a mudra.
    """
    def __init__(self):
        self.state = "NO_HAND"
        self.current_mudra = None
        self.method = None
        self.confidence = 0.0

        self.enter_count = 0
        self.exit_count = 0
        self.mismatch_count = 0  # ISSUE 3 FIX: tolerance for noisy frames

        # Tunable thresholds
        self.ENTER_THRESHOLD = 3   # frames to confirm entry
        self.EXIT_THRESHOLD = 2    # frames to confirm exit
        self.MAX_MISMATCH = 1      # ISSUE 3 FIX: allow 1 noisy frame during entry

    def reset(self):
        """Reset to HAND_DETECTED state (hand present but no mudra)."""
        self.state = "HAND_DETECTED"
        self.current_mudra = None
        self.method = None
        self.confidence = 0.0
        self.enter_count = 0
        self.exit_count = 0
        self.mismatch_count = 0

    def update(self, hand_present, candidate_name, candidate_conf, candidate_method):
        """
        Update FSM with latest hybrid detection result.

        Inputs:
            hand_present     : bool
            candidate_name   : str or None
            candidate_conf   : float
            candidate_method : "RULE" | "ML" | None

        Returns:
            (display_name, display_conf, display_method)
        """

        # ---------------------------
        # NO HAND
        # ---------------------------
        if not hand_present:
            self.state = "NO_HAND"
            self.current_mudra = None
            return ("No hand detected", 0.0, "")

        # ---------------------------
        # HAND DETECTED
        # ---------------------------
        if self.state == "NO_HAND":
            self.state = "HAND_DETECTED"

        if self.state == "HAND_DETECTED":
            if candidate_name is None:
                return ("Show mudra...", 0.0, "")
            
            # RULE shortcut → immediate confirmation
            if candidate_method == "RULE":
                self.state = "CONFIRMED_MUDRA"
                self.current_mudra = candidate_name
                self.method = "RULE"
                self.confidence = 1.0
                return (self.current_mudra, self.confidence, self.method)

            # ML candidate → debounce
            self.state = "ENTERING_MUDRA"
            self.current_mudra = candidate_name
            self.method = candidate_method
            self.confidence = candidate_conf
            self.enter_count = 1
            return ("Detecting...", 0.0, "")

        # ---------------------------
        # ENTERING
        # ---------------------------
        if self.state == "ENTERING_MUDRA":
            if candidate_name == self.current_mudra:
                self.enter_count += 1
                self.mismatch_count = 0  # Reset mismatch on successful match
                self.confidence = candidate_conf

                if self.enter_count >= self.ENTER_THRESHOLD:
                    self.state = "CONFIRMED_MUDRA"
                    return (self.current_mudra, self.confidence, self.method)
                else:
                    return ("Detecting...", 0.0, "")
            else:
                # ISSUE 3 FIX: Allow some noise tolerance during entry
                self.mismatch_count += 1
                if self.mismatch_count > self.MAX_MISMATCH:
                    self.reset()
                    return ("Stabilizing...", 0.0, "")
                # Keep trying - one noisy frame is tolerated
                return ("Detecting...", 0.0, "")

        # ---------------------------
        # CONFIRMED
        # ---------------------------
        if self.state == "CONFIRMED_MUDRA":
            if candidate_name == self.current_mudra:
                self.exit_count = 0
                # ISSUE 1 FIX: Don't downgrade RULE confidence with ML confidence
                if self.method != "RULE":
                    self.confidence = candidate_conf
                return (self.current_mudra, self.confidence, self.method)
            else:
                self.state = "EXITING_MUDRA"
                self.exit_count = 1
                return (self.current_mudra, self.confidence, self.method)

        # ---------------------------
        # EXITING
        # ---------------------------
        if self.state == "EXITING_MUDRA":
            if candidate_name == self.current_mudra:
                self.state = "CONFIRMED_MUDRA"
                self.exit_count = 0
                return (self.current_mudra, self.confidence, self.method)
            else:
                self.exit_count += 1
                if self.exit_count >= self.EXIT_THRESHOLD:
                    self.reset()
                    return ("Show mudra...", 0.0, "")
                else:
                    return (self.current_mudra, self.confidence, self.method)

        # Fallback (should never reach here)
        return ("Show mudra...", 0.0, "")


# ============================================================
# CONFIGURATION
# ============================================================
DEBUG = False
CAMERA_INDEX = 0

# Frame processing
FRAME_SKIP = 2               # Process every Nth frame

# ML confidence threshold
ML_CONF_THRESHOLD = 0.55     # ML prediction must exceed this confidence

# Stability parameters
STABILITY_DISTANCE_THRESHOLD = 0.02  # Maximum movement allowed for "steady" hand

# ============================================================
# LOAD MACHINE LEARNING MODEL
# ============================================================
try:
    model_path = os.path.join(os.path.dirname(__file__), "mudra_rf_model.pkl")
    model = joblib.load(model_path)
    model_classes = model.classes_
    print(f"✓ ML Model loaded with {len(model_classes)} classes")
except Exception as e:
    print(f"✗ Failed to load ML model: {e}")
    # Don't exit here if imported by api_server (let api_server handle it or crash later)
    # But for now, exit(1) is fine as api_server imports it at top level
    exit(1)

# ============================================================
# MEDIAPIPE SETUP
# ============================================================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ============================================================
# GLOBAL STATE
# ============================================================
frame_count = 0
prev_landmarks = None
mudra_fsm = MudraFSM()  # FSM replaces voting queue

# ============================================================
# UTILITY FUNCTIONS (Distance & Geometry)
# ============================================================

def get_distance(a, b):
    """Euclidean distance between two landmarks (optimized with native math.dist)."""
    return math.dist((a.x, a.y), (b.x, b.y))


def get_scale_ref(landmarks):
    """Palm width reference: distance from wrist (0) to middle MCP (9)."""
    return get_distance(landmarks[0], landmarks[9]) + 1e-6


# ISSUE 4 FIX: Removed misleading compute_distance_tables() - we use lazy computation only


def compute_distance_on_demand(landmarks, i, j, scale_ref):
    """Compute normalized distance between two landmarks on-demand."""
    d = math.sqrt(
        (landmarks[i].x - landmarks[j].x) ** 2 +
        (landmarks[i].y - landmarks[j].y) ** 2
    )
    return d / scale_ref


def norm_dist_lazy(landmarks, i, j, scale_ref):
    """Lazy normalized distance calculation."""
    return compute_distance_on_demand(landmarks, i, j, scale_ref)


def get_angle(a, b, c, landmarks):
    """
    Calculate angle (in degrees) at point b formed by a-b-c.
    Works with MediaPipe landmark objects.
    """
    try:
        ax, ay = landmarks[a].x, landmarks[a].y
        bx, by = landmarks[b].x, landmarks[b].y
        cx, cy = landmarks[c].x, landmarks[c].y

        v1 = (ax - bx, ay - by)
        v2 = (cx - bx, cy - by)

        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.hypot(*v1)
        mag2 = math.hypot(*v2)

        if mag1 == 0 or mag2 == 0:
            return 180

        cosA = max(-1, min(1, dot / (mag1 * mag2)))
        return math.degrees(math.acos(cosA))
    except Exception:
        return 180


def angle_between(v1, v2):
    """
    Calculate angle (in degrees) between two 2D vectors.
    Returns None if either vector has zero magnitude.
    """
    mag1 = math.hypot(v1[0], v1[1])
    mag2 = math.hypot(v2[0], v2[1])
    
    if mag1 == 0 or mag2 == 0:
        return None
    
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    cosA = max(-1, min(1, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cosA))


def is_finger_straight(landmarks, mcp, pip, tip, threshold=0.9):
    """
    Check if finger is straight using straightness ratio:
    ratio = (MCP→TIP) / (MCP→PIP + PIP→TIP)
    """
    scale_ref = get_scale_ref(landmarks)
    mcp_pip = norm_dist_lazy(landmarks, mcp, pip, scale_ref)
    pip_tip = norm_dist_lazy(landmarks, pip, tip, scale_ref)
    mcp_tip = norm_dist_lazy(landmarks, mcp, tip, scale_ref)

    total = mcp_pip + pip_tip
    if total == 0:
        return False
    
    straightness_ratio = mcp_tip / total
    return straightness_ratio > threshold


def is_hand_steady(landmarks, prev_landmarks, threshold=0.02):
    """
    Check if hand is steady by comparing key landmarks with previous frame.
    Returns True if hand movement is below threshold.
    Checks wrist + ALL fingertips to catch any finger movement.
    """
    if prev_landmarks is None:
        return False
    
    # Check Wrist + ALL Fingertips (thumb, index, middle, ring, pinky)
    # This ensures we catch movement even if only one finger moves
    key_indices = [0, 4, 8, 12, 16, 20]
    
    for idx in key_indices:
        curr = landmarks[idx]
        prev = prev_landmarks[idx]
        
        # Optimized: Use native math.dist for 3D distance (faster C implementation)
        distance = math.dist((curr.x, curr.y, curr.z), (prev.x, prev.y, prev.z))
        
        if distance > threshold:
            return False
    
    return True


# ============================================================
# RULE-BASED MUDRA DETECTION FUNCTIONS
# ============================================================
# These functions return True if the mudra is detected, False otherwise.
# Each function implements strict geometric rules for a specific mudra.
# ============================================================

def is_pataka_mudra(landmarks, scale_ref):
    """All four fingers straight and close together, thumb tucked."""
    index_straight = is_finger_straight(landmarks, 5, 6, 8, 0.97)
    middle_straight = is_finger_straight(landmarks, 9, 10, 12, 0.97)
    ring_straight = is_finger_straight(landmarks, 13, 14, 16, 0.97)
    pinky_straight = is_finger_straight(landmarks, 17, 18, 20, 0.97)
    
    if not (index_straight and middle_straight and ring_straight and pinky_straight):
        return False
    
    # Thumb should be tucked near index base
    thumb_index_nd = norm_dist_lazy(landmarks, 4, 5, scale_ref)
    mcp_nd = norm_dist_lazy(landmarks, 5, 9, scale_ref)
    thumb_tucked = thumb_index_nd < (mcp_nd * 1.5)
    
    return thumb_tucked


def is_tripataka_mudra(landmarks, scale):
    """Index, middle, pinky straight; ring bent; thumb tucked."""
    if not is_finger_straight(landmarks, 5, 6, 8, 0.94): 
        return False
    if not is_finger_straight(landmarks, 9, 10, 12, 0.94): 
        return False
    if not is_finger_straight(landmarks, 17, 18, 20, 0.94): 
        return False

    # Ring must be bent
    if is_finger_straight(landmarks, 13, 14, 16, 0.88):
        return False

    # Thumb MUST NOT touch ring tip
    if norm_dist_lazy(landmarks, 4, 16, scale) < scale * 0.9:
        return False

    # Thumb should be tucked
    if norm_dist_lazy(landmarks, 4, 5, scale) > scale * 1.6:
        return False

    return True


def is_musthi_mudra(landmarks, scale_ref):
    """Fist: all fingers bent, thumb on top or tucked."""
    index_bent = not is_finger_straight(landmarks, 5, 6, 8, 0.8)
    middle_bent = not is_finger_straight(landmarks, 9, 10, 12, 0.8)
    ring_bent = not is_finger_straight(landmarks, 13, 14, 16, 0.8)
    pinky_bent = not is_finger_straight(landmarks, 17, 18, 20, 0.8)
    
    if not (index_bent and middle_bent and ring_bent and pinky_bent):
        return False
    
    # Thumb touches bent fingers
    thumb_index_pip_nd = norm_dist_lazy(landmarks, 4, 6, scale_ref)
    thumb_middle_pip_nd = norm_dist_lazy(landmarks, 4, 10, scale_ref)
    mcp_nd = norm_dist_lazy(landmarks, 5, 9, scale_ref)
    touch_threshold = mcp_nd * 1.5
    
    return (thumb_index_pip_nd < touch_threshold) or (thumb_middle_pip_nd < touch_threshold)


def is_suchi_mudra(landmarks, scale_ref):
    """Index finger pointing up, others bent, thumb touches middle/ring."""
    # Index must be very straight
    pip = get_angle(5, 6, 7, landmarks)
    dip = get_angle(6, 7, 8, landmarks)
    if not (pip > 170 and dip > 170):
        return False

    # Index tip extended away from wrist
    wrist = landmarks[0]
    tip = landmarks[8]
    mcp = landmarks[5]
    if math.dist((tip.x, tip.y), (wrist.x, wrist.y)) <= math.dist((mcp.x, mcp.y), (wrist.x, wrist.y)):
        return False

    # Other fingers must be bent
    if is_finger_straight(landmarks, 9, 10, 12, 0.75):
        return False
    if is_finger_straight(landmarks, 13, 14, 16, 0.75):
        return False
    if is_finger_straight(landmarks, 17, 18, 20, 0.75):
        return False

    # Thumb touches middle or ring
    ref = norm_dist_lazy(landmarks, 5, 9, scale_ref)
    if ref == 0:
        return False

    t_mid1 = norm_dist_lazy(landmarks, 4, 10, scale_ref)
    t_mid2 = norm_dist_lazy(landmarks, 4, 11, scale_ref)
    t_ring1 = norm_dist_lazy(landmarks, 4, 14, scale_ref)
    t_ring2 = norm_dist_lazy(landmarks, 4, 15, scale_ref)
    touch_thresh = ref * 1.6
    
    touches_mid_ring = (
        t_mid1 < touch_thresh or
        t_mid2 < touch_thresh or
        t_ring1 < touch_thresh or
        t_ring2 < touch_thresh
    )
    
    if not touches_mid_ring:
        return False

    # Thumb must NOT touch index
    if norm_dist_lazy(landmarks, 4, 8, scale_ref) < ref * 2.0:
        return False

    return True


def is_arala_mudra(landmarks, scale_ref):
    """Middle, ring, pinky straight and close; index bent; thumb straight."""
    # Index must be bent
    index_straight = is_finger_straight(landmarks, 5, 6, 8, 0.85)
    if index_straight:
        return False

    # Middle, ring, pinky must be very straight
    middle_straight = is_finger_straight(landmarks, 9, 10, 12, 0.97)
    ring_straight = is_finger_straight(landmarks, 13, 14, 16, 0.97)
    pinky_straight = is_finger_straight(landmarks, 17, 18, 20, 0.97)
    
    if not (middle_straight and ring_straight and pinky_straight):
        return False

    # Middle, ring, pinky must be close together
    d_mr = norm_dist_lazy(landmarks, 12, 16, scale_ref)
    d_rp = norm_dist_lazy(landmarks, 16, 20, scale_ref)
    d_mp = norm_dist_lazy(landmarks, 12, 20, scale_ref)
    close_thresh = scale_ref * 1.2
    
    if not (d_mr < close_thresh and d_rp < close_thresh and d_mp < close_thresh):
        return False

    # Thumb must be straight
    thumb_straight = is_finger_straight(landmarks, 2, 3, 4, 0.93)
    if not thumb_straight:
        return False

    return True


def is_hamsasya_mudra(landmarks, scale_ref):
    """Thumb tip touches index tip; index bent; middle, ring, pinky straight."""
    # Thumb tip & Index tip must be VERY close
    d_thumb_index_tip = norm_dist_lazy(landmarks, 4, 8, scale_ref)
    if d_thumb_index_tip > 0.28 * scale_ref:
        return False

    # Index must be clearly bent
    index_is_straight = is_finger_straight(landmarks, 5, 6, 8, 0.94)
    if index_is_straight:
        return False

    # Middle, Ring, Pinky must be straight
    middle_straight = is_finger_straight(landmarks, 9, 10, 12, 0.93)
    ring_straight = is_finger_straight(landmarks, 13, 14, 16, 0.92)
    pinky_straight = is_finger_straight(landmarks, 17, 18, 20, 0.90)

    if not (middle_straight and ring_straight and pinky_straight):
        return False

    return True


def is_shikharam_mudra(landmarks, scale_ref):
    """Fist with thumb pointing up."""
    # All fingers bent
    index_bent = not is_finger_straight(landmarks, 5, 6, 8, 0.8)
    middle_bent = not is_finger_straight(landmarks, 9, 10, 12, 0.8)
    ring_bent = not is_finger_straight(landmarks, 13, 14, 16, 0.8)
    pinky_bent = not is_finger_straight(landmarks, 17, 18, 20, 0.8)
    
    if not (index_bent and middle_bent and ring_bent and pinky_bent):
        return False
    
    # Thumb must be straight
    thumb_straight = is_finger_straight(landmarks, 2, 3, 4, 0.85)
    if not thumb_straight:
        return False
    
    # Thumb should NOT be tucked
    thumb_index_nd = norm_dist_lazy(landmarks, 4, 6, scale_ref)
    thumb_middle_nd = norm_dist_lazy(landmarks, 4, 10, scale_ref)
    mcp_nd = norm_dist_lazy(landmarks, 5, 9, scale_ref)
    touch_threshold = mcp_nd * 1.5
    thumb_is_tucked = (thumb_index_nd < touch_threshold or thumb_middle_nd < touch_threshold)
    
    return not thumb_is_tucked


def is_kartari_mukham_mudra(landmarks, scale_ref):
    """Index and middle fingers straight and extended; ring and pinky bent; thumb touches ring finger."""
    index_straight = is_finger_straight(landmarks, 5, 6, 8)
    middle_straight = is_finger_straight(landmarks, 9, 10, 12)
    if not (index_straight and middle_straight):
        return False

    index_tip_wrist_nd = norm_dist_lazy(landmarks, 8, 0, scale_ref)
    index_mcp_wrist_nd = norm_dist_lazy(landmarks, 5, 0, scale_ref)
    middle_tip_wrist_nd = norm_dist_lazy(landmarks, 12, 0, scale_ref)
    middle_mcp_wrist_nd = norm_dist_lazy(landmarks, 9, 0, scale_ref)
    if not (index_tip_wrist_nd > index_mcp_wrist_nd and
            middle_tip_wrist_nd > middle_mcp_wrist_nd):
        return False

    ring_bent = not is_finger_straight(landmarks, 13, 14, 16, 0.8)
    pinky_bent = not is_finger_straight(landmarks, 17, 18, 20, 0.8)
    if not (ring_bent and pinky_bent):
        return False

    thumb_ring_pip_nd = norm_dist_lazy(landmarks, 4, 14, scale_ref)
    thumb_ring_tip_nd = norm_dist_lazy(landmarks, 4, 16, scale_ref)
    mcp_nd = norm_dist_lazy(landmarks, 5, 9, scale_ref)
    touch_threshold = mcp_nd * 2.0
    return (thumb_ring_pip_nd < touch_threshold) or (thumb_ring_tip_nd < touch_threshold)


def is_sarpashirsha_mudra(landmarks, scale_ref):
    """All four fingers straight and converging at tips; thumb tucked."""
    loose = 0.80
    idx = is_finger_straight(landmarks, 5, 6, 8, loose)
    mid = is_finger_straight(landmarks, 9, 10, 12, loose)
    rng = is_finger_straight(landmarks, 13, 14, 16, loose)
    pnk = is_finger_straight(landmarks, 17, 18, 20, loose)
    if not (idx and mid and rng and pnk):
        return False
    tip_dist = norm_dist_lazy(landmarks, 8, 20, scale_ref)
    mcp_dist = norm_dist_lazy(landmarks, 5, 17, scale_ref)
    if not (tip_dist < mcp_dist):
        return False
    thumb_index_nd = norm_dist_lazy(landmarks, 4, 5, scale_ref)
    mcp_ref_nd = norm_dist_lazy(landmarks, 5, 9, scale_ref)
    return thumb_index_nd < (mcp_ref_nd * 1.5)


def is_chatura_mudra(landmarks, scale_ref):
    """Index, middle, ring straight and close; pinky straight; thumb tucked deep inside palm."""
    if not (
        is_finger_straight(landmarks, 5, 6, 8, 0.85) and
        is_finger_straight(landmarks, 9, 10, 12, 0.85) and
        is_finger_straight(landmarks, 13, 14, 16, 0.85) and
        is_finger_straight(landmarks, 17, 18, 20, 0.85)
    ):
        return False

    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]

    d_im = math.dist((index_tip.x, index_tip.y), (middle_tip.x, middle_tip.y))
    d_mr = math.dist((middle_tip.x, middle_tip.y), (ring_tip.x, ring_tip.y))

    if d_im > scale_ref * 0.30 or d_mr > scale_ref * 0.30:
        return False

    wrist = landmarks[0]
    middle_mcp = (landmarks[9].x, landmarks[9].y)
    palm_center = ((wrist.x + middle_mcp[0]) / 2, (wrist.y + middle_mcp[1]) / 2)

    thumb_tip = (landmarks[4].x, landmarks[4].y)
    index_mcp = (landmarks[5].x, landmarks[5].y)
    ring_mcp = (landmarks[13].x, landmarks[13].y)
    pinky_mcp = (landmarks[17].x, landmarks[17].y)

    d_thumb_palm = math.dist(thumb_tip, palm_center)
    d_thumb_index = math.dist(thumb_tip, (landmarks[8].x, landmarks[8].y))
    d_thumb_middle = math.dist(thumb_tip, (landmarks[12].x, landmarks[12].y))
    d_thumb_ring = math.dist(thumb_tip, (landmarks[16].x, landmarks[16].y))
    d_thumb_pinky = math.dist(thumb_tip, (landmarks[20].x, landmarks[20].y))

    tolerance = scale_ref * 0.01
    thumb_depth_ok = thumb_tip[1] > (middle_mcp[1] - tolerance)
    thumb_x_inside = min(index_mcp[0], pinky_mcp[0]) < thumb_tip[0] < max(index_mcp[0], pinky_mcp[0])
    thumb_deep_inside = (
        d_thumb_palm < d_thumb_index and
        d_thumb_palm < d_thumb_middle and
        d_thumb_palm < d_thumb_ring and
        d_thumb_palm < d_thumb_pinky
    )

    if not (thumb_depth_ok and thumb_deep_inside and thumb_x_inside):
        return False

    return True


def is_trishula_mudra(landmarks, scale_ref):
    """Index, middle, ring straight; pinky bent; thumb touches pinky."""
    # Index, middle, ring must be straight
    index_straight = is_finger_straight(landmarks, 5, 6, 8)
    middle_straight = is_finger_straight(landmarks, 9, 10, 12)
    ring_straight = is_finger_straight(landmarks, 13, 14, 16)
    if not (index_straight and middle_straight and ring_straight):
        return False
    
    # Pinky must be bent
    pinky_bent = not is_finger_straight(landmarks, 17, 18, 20, 0.8)
    if not pinky_bent:
        return False
    
    # Thumb touches pinky (or is close to it)
    thumb_pinky_nd = norm_dist_lazy(landmarks, 4, 20, scale_ref)
    mcp_nd = norm_dist_lazy(landmarks, 5, 9, scale_ref)
    return thumb_pinky_nd < (mcp_nd * 1.0)


def is_mrigasheersha_mudra(landmarks, scale_ref):
    """Pinky and thumb straight; index, middle, ring bent; thumb angled outward."""
    # Pinky and thumb must be straight
    pinky_straight = is_finger_straight(landmarks, 17, 18, 20, 0.85)
    thumb_straight = is_finger_straight(landmarks, 2, 3, 4, 0.80)
    
    # Index, middle, ring must be bent
    index_bent = not is_finger_straight(landmarks, 5, 6, 8, 0.85)
    middle_bent = not is_finger_straight(landmarks, 9, 10, 12, 0.85)
    ring_bent = not is_finger_straight(landmarks, 13, 14, 16, 0.85)
    
    if not (pinky_straight and thumb_straight and index_bent and middle_bent and ring_bent):
        return False
    
    # Thumb must be angled outward (>35 degrees from hand direction)
    v_thumb = (landmarks[4].x - landmarks[2].x, landmarks[4].y - landmarks[2].y)
    v_hand = (landmarks[9].x - landmarks[0].x, landmarks[9].y - landmarks[0].y)
    ang = angle_between(v_thumb, v_hand)
    
    if ang is None:
        return False
    
    return ang > 35


def is_simhamukha_mudra(landmarks, scale_ref):
    """Index and pinky straight; thumb, middle, ring tips clustered together."""
    ref = norm_dist_lazy(landmarks, 5, 9, scale_ref)
    if ref == 0:
        return False
    
    # Index and pinky must be straight
    if not is_finger_straight(landmarks, 5, 6, 8, 0.90):
        return False
    if not is_finger_straight(landmarks, 17, 18, 20, 0.90):
        return False
    
    # Thumb, middle, ring tips must be clustered together
    d_tm = norm_dist_lazy(landmarks, 4, 12, scale_ref)
    d_tr = norm_dist_lazy(landmarks, 4, 16, scale_ref)
    d_mr = norm_dist_lazy(landmarks, 12, 16, scale_ref)
    cluster_thresh = ref * 1.9
    
    if not (d_tm < cluster_thresh and d_tr < cluster_thresh and d_mr < cluster_thresh):
        return False
    
    # Middle and ring should not be fully extended upward
    mid_straight = is_finger_straight(landmarks, 9, 10, 12, 0.92)
    ring_straight = is_finger_straight(landmarks, 13, 14, 16, 0.92)
    
    if mid_straight and landmarks[12].y < landmarks[9].y:
        return False
    if ring_straight and landmarks[16].y < landmarks[13].y:
        return False
    
    return True


def is_ardhapataka_mudra(landmarks, scale_ref):
    """Index and middle straight; ring and pinky bent; thumb tucked."""
    # Index and middle must be straight
    index_straight = is_finger_straight(landmarks, 5, 6, 8)
    middle_straight = is_finger_straight(landmarks, 9, 10, 12)
    if not (index_straight and middle_straight):
        return False
    
    # Ring and pinky must be bent
    ring_bent = not is_finger_straight(landmarks, 13, 14, 16, 0.8)
    pinky_bent = not is_finger_straight(landmarks, 17, 18, 20, 0.8)
    if not (ring_bent and pinky_bent):
        return False
    
    # Thumb should be tucked near index base
    thumb_index_nd = norm_dist_lazy(landmarks, 4, 5, scale_ref)
    mcp_nd = norm_dist_lazy(landmarks, 5, 9, scale_ref)
    return thumb_index_nd < (mcp_nd * 1.5)


def is_mayura_mudra(landmarks, scale_ref):
    """Index, middle, pinky straight; ring bent; thumb touches ring tip."""
    # Index, middle, pinky must be straight
    if not is_finger_straight(landmarks, 5, 6, 8, 0.90):
        return False
    if not is_finger_straight(landmarks, 9, 10, 12, 0.90):
        return False
    if not is_finger_straight(landmarks, 17, 18, 20, 0.90):
        return False
    
    # Ring MUST be bent
    if is_finger_straight(landmarks, 13, 14, 16, 0.88):
        return False
    
    # Thumb tip MUST TOUCH ring tip (STRICT - reduced threshold for actual contact)
    thumb_ring_nd = norm_dist_lazy(landmarks, 4, 16, scale_ref)
    if thumb_ring_nd > scale_ref * 0.45:
        return False
    
    return True


def is_shuka_tundam_mudra(landmarks, scale_ref):
    """Middle and pinky straight; index and ring bent; thumb tucked."""
    # Middle and pinky must be straight
    middle_straight = is_finger_straight(landmarks, 9, 10, 12)
    pinky_straight = is_finger_straight(landmarks, 17, 18, 20)
    
    # Index and ring must be bent
    index_bent = not is_finger_straight(landmarks, 5, 6, 8, 0.85)
    ring_bent = not is_finger_straight(landmarks, 13, 14, 16, 0.85)
    
    if not (middle_straight and pinky_straight and index_bent and ring_bent):
        return False
    
    # Thumb should be tucked near index base
    thumb_index_nd = norm_dist_lazy(landmarks, 4, 5, scale_ref)
    mcp_nd = norm_dist_lazy(landmarks, 5, 9, scale_ref)
    return thumb_index_nd < (mcp_nd * 1.5)


# ============================================================
# RULE-BASED MUDRA REGISTRY
# ============================================================
# Order = Priority (higher priority mudras checked first)
# ============================================================

RULE_MUDRA_FUNCTIONS = {
    # High priority (very distinctive geometric patterns)
    "Musthi Mudra": is_musthi_mudra,
    "Suchi Mudra": is_suchi_mudra,
    "Shikharam Mudra": is_shikharam_mudra,
    "Hamsasya Mudra": is_hamsasya_mudra,
    "Mayura Mudra": is_mayura_mudra,
    "Tripataka Mudra": is_tripataka_mudra,
    "Kartari Mukham Mudra": is_kartari_mukham_mudra,
    "Trishula Mudra": is_trishula_mudra,
    "Mrigasheersha Mudra": is_mrigasheersha_mudra,
    "Simhamukha Mudra": is_simhamukha_mudra,
    "Ardhapataka Mudra": is_ardhapataka_mudra,
    "Shuka Tundam Mudra": is_shuka_tundam_mudra,
    
    # Medium priority
    "Arala Mudra": is_arala_mudra,
    "Pataka Mudra": is_pataka_mudra,
    "Sarpashirsha Mudra": is_sarpashirsha_mudra,
    "Chatura Mudra": is_chatura_mudra,
}


# ============================================================
# MACHINE LEARNING FEATURE EXTRACTION
# ============================================================

def extract_ml_features(landmarks, scale_ref):
    """
    Extract 17 features for ML model:
    - 5 finger straightness values
    - 4 thumb-to-fingertip distances
    - 3 fingertip clustering distances
    - 4 joint angles
    - 1 palm orientation (Z-axis)
    """
    features = []

    # 1️⃣ Finger straightness (5 features)
    def finger_straightness(mcp, pip, tip):
        d1 = math.dist((landmarks[mcp].x, landmarks[mcp].y),
                       (landmarks[pip].x, landmarks[pip].y))
        d2 = math.dist((landmarks[pip].x, landmarks[pip].y),
                       (landmarks[tip].x, landmarks[tip].y))
        d3 = math.dist((landmarks[mcp].x, landmarks[mcp].y),
                       (landmarks[tip].x, landmarks[tip].y))
        return d3 / (d1 + d2 + 1e-6)

    features += [
        finger_straightness(2, 3, 4),    # Thumb
        finger_straightness(5, 6, 8),    # Index
        finger_straightness(9, 10, 12),  # Middle
        finger_straightness(13, 14, 16), # Ring
        finger_straightness(17, 18, 20)  # Pinky
    ]

    # 2️⃣ Thumb-to-fingertip distances (4 features)
    def norm_dist_ml(i, j):
        return math.dist(
            (landmarks[i].x, landmarks[i].y),
            (landmarks[j].x, landmarks[j].y)
        ) / scale_ref

    features += [
        norm_dist_ml(4, 8),   # Thumb to index
        norm_dist_ml(4, 12),  # Thumb to middle
        norm_dist_ml(4, 16),  # Thumb to ring
        norm_dist_ml(4, 20)   # Thumb to pinky
    ]

    # 3️⃣ Fingertip clustering (3 features)
    features += [
        norm_dist_ml(8, 12),  # Index to middle
        norm_dist_ml(12, 16), # Middle to ring
        norm_dist_ml(16, 20)  # Ring to pinky
    ]

    # 4️⃣ Joint angles (4 features)
    features += [
        get_angle(5, 6, 7, landmarks),   # Index PIP
        get_angle(6, 7, 8, landmarks),   # Index DIP
        get_angle(9, 10, 11, landmarks), # Middle PIP
        get_angle(13, 14, 15, landmarks) # Ring PIP
    ]

    # 5️⃣ Palm orientation (Z-axis) (1 feature)
    features.append(abs(landmarks[9].z - landmarks[0].z) / scale_ref)

    return np.array(features, dtype=np.float32)


# ============================================================
# HAND NORMALIZATION (Left → Right) - REMOVED FOR ML COMPATIBILITY
# ============================================================

# Normalization removed - ML model was trained on raw landmark data
# def normalize_hand_orientation(landmarks, handedness_label):
#     """Normalization disabled - using raw data as ML model expects"""
#     pass


# ============================================================
# HYBRID DECISION PIPELINE
# ============================================================

def detect_mudra_hybrid(landmarks, handedness_label, prev_landmarks):
    """
    HYBRID DECISION LOGIC (OPTIMIZED):
    
    1. Check hand stability (only run ML if hand is steady)
    2. Run rule-based checks (strict geometric patterns)
    3. IF rule confidently matches (score = 1.0):
           Return rule result
    4. ELSE:
           IF hand is steady:
                Extract ML features
                Run ML model
                IF ML confidence ≥ threshold:
                    Return ML result
                ELSE:
                    Return "Unknown"
           ELSE:
                Return "Stabilizing..." (hand moving)
    
    Returns: (mudra_name, confidence, method)
        - mudra_name: detected mudra or "Unknown"
        - confidence: 1.0 for rules, ML probability for ML
        - method: "RULE" or "ML"
    """
    
    # NO normalization - use raw data as ML model expects
    
    # Get scale reference for normalized distance calculations
    scale_ref = get_scale_ref(landmarks)
    
    # Step 1: Try rule-based detection (priority order) - always fast
    for mudra_name, check_func in RULE_MUDRA_FUNCTIONS.items():
        try:
            if check_func(landmarks, scale_ref):
                # Rule match found - return immediately with full confidence
                return (mudra_name, 1.0, "RULE")
        except Exception as e:
            if DEBUG:
                print(f"Rule error [{mudra_name}]: {e}")
            continue
    
    # Step 2: Check if hand is steady before running ML
    if not is_hand_steady(landmarks, prev_landmarks, STABILITY_DISTANCE_THRESHOLD):
        # ISSUE 2 FIX: Return None for method - FSM doesn't need to know why
        return ("Stabilizing...", 0.0, None)
    
    # Step 3: No rule match and hand is steady - try ML prediction
    try:
        # Extract ML features
        features = extract_ml_features(landmarks, scale_ref)
        X = features.reshape(1, -1)
        
        # Get ML prediction with probabilities
        probs = model.predict_proba(X)[0]
        max_idx = np.argmax(probs)
        ml_conf = probs[max_idx]
        ml_pred = model_classes[max_idx]
        
        # Check ML confidence threshold
        if ml_conf >= ML_CONF_THRESHOLD:
            return (ml_pred, ml_conf, "ML")
        else:
            return ("Unknown", ml_conf, "ML")
            
    except Exception as e:
        if DEBUG:
            print(f"ML prediction error: {e}")
        return ("Unknown", 0.0, "ML")


# ============================================================
# MAIN WEBCAM LOOP
# ============================================================

def main():
    """
    Main webcam loop with hybrid mudra detection using FSM.
    """
    global frame_count, prev_landmarks, DEBUG, mudra_fsm
    
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("✗ Error: Could not open webcam")
        exit(1)
    
    print("\n" + "="*70)
    print("HYBRID MUDRA DETECTION SYSTEM")
    print("="*70)
    print("Controls:")
    print("  Q - Quit")
    print("  D - Toggle debug mode")
    print("="*70 + "\n")
    
    # FPS calculation
    prev_time = cv2.getTickCount()
    
    # STICKY DISPLAY VARIABLES (persist across frames to prevent flickering)
    display_text = "Show mudra..."
    display_color = (0, 165, 255)  # Orange
    method_text = ""
    conf_text = ""
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("✗ Failed to read frame")
            break
        
        # Resize for performance
        frame = cv2.resize(frame, (640, 480))
        frame = cv2.flip(frame, 1)  # Mirror for user experience
        
        frame_count += 1
        # Display variables are now sticky - only updated when detection runs
        
        # Process frame with MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = hands.process(frame_rgb)
        frame_rgb.flags.writeable = True
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = hand_landmarks.landmark
            
            # Draw hand landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
            
            # Get handedness label
            handedness_label = "Right"  # Default
            if results.multi_handedness:
                handedness_label = results.multi_handedness[0].classification[0].label
            
            # Run hybrid detection (only every FRAME_SKIP frames for performance)
            if frame_count % FRAME_SKIP == 0:
                mudra_name, confidence, method = detect_mudra_hybrid(
                    landmarks, handedness_label, prev_landmarks
                )
                
                # Normalize candidate output for FSM
                if mudra_name in ["Unknown", "Stabilizing..."]:
                    candidate_name = None
                    candidate_conf = 0.0
                    candidate_method = None
                else:
                    candidate_name = mudra_name
                    candidate_conf = confidence
                    candidate_method = method
                
                # Update FSM with detection result
                display_text, disp_conf, disp_method = mudra_fsm.update(
                    hand_present=True,
                    candidate_name=candidate_name,
                    candidate_conf=candidate_conf,
                    candidate_method=candidate_method
                )
                
                # Update UI variables based on FSM output
                if disp_method:
                    conf_text = f"Conf: {disp_conf*100:.1f}%"
                    method_text = f"Method: {disp_method}"
                    display_color = (0, 255, 0)  # Green
                else:
                    conf_text = ""
                    method_text = ""
                    display_color = (0, 165, 255)  # Orange
                
                # Update previous landmarks for stability check
                prev_landmarks = [SimpleNamespace(x=lm.x, y=lm.y, z=lm.z) for lm in landmarks]
        
        else:
            # No hand detected - update FSM
            display_text, _, _ = mudra_fsm.update(
                hand_present=False,
                candidate_name=None,
                candidate_conf=0.0,
                candidate_method=None
            )
            display_color = (0, 0, 255)  # Red
            conf_text = ""
            method_text = ""
        
        # Calculate FPS
        current_time = cv2.getTickCount()
        time_diff = (current_time - prev_time) / cv2.getTickFrequency()
        fps = 1.0 / time_diff if time_diff > 0 else 0
        prev_time = current_time
        
        # ============================================================
        # DISPLAY OVERLAY
        # ============================================================
        
        # Main prediction text
        cv2.putText(frame, display_text, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, display_color, 3, cv2.LINE_AA)
        
        # Confidence text
        if conf_text:
            cv2.putText(frame, conf_text, (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
        
        # Method text
        if method_text:
            cv2.putText(frame, method_text, (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # FPS counter
        cv2.putText(frame, f"FPS: {int(fps)}", (540, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Debug info - show FSM state
        if DEBUG:
            debug_info = f"State: {mudra_fsm.state} | Enter: {mudra_fsm.enter_count} | Exit: {mudra_fsm.exit_count}"
            cv2.putText(frame, debug_info, (20, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        
        # Mode indicator
        mode_text = "HYBRID MODE (FSM) - Rules → ML"
        cv2.putText(frame, mode_text, (20, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        
        # Show frame
        cv2.imshow("Hybrid Mudra Detection", frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('d') or key == ord('D'):
            DEBUG = not DEBUG
            print(f"Debug mode: {'ON' if DEBUG else 'OFF'}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Hybrid mudra detection system closed")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
