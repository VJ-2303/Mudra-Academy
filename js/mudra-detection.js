/**
 * ================================================================================
 * MUDRA FSM & RULE-BASED DETECTION - JavaScript
 * ================================================================================
 * Ported from Python hybrid_webcam.py for offline JavaScript execution.
 * Combines rule-based logic (strict geometric checks) with ML prediction
 * using Finite State Machine for flicker-free output.
 * ================================================================================
 */

/**
 * Finite State Machine for robust mudra detection with hysteresis.
 * Eliminates flickering by requiring multiple consecutive frames
 * before confirming entry/exit of a mudra.
 */
class MudraFSM {
    constructor() {
        this.state = "NO_HAND";
        this.currentMudra = null;
        this.method = null;
        this.confidence = 0.0;

        this.enterCount = 0;
        this.exitCount = 0;
        this.mismatchCount = 0;

        // Tunable thresholds (matching Python API server for consistency)
        // REDUCED for faster response as per user feedback
        this.ENTER_THRESHOLD = 1;  // Almost instant entry (was 3)
        this.EXIT_THRESHOLD = 5;   // Keep exit threshold high for stability
        this.MAX_MISMATCH = 3;     // More forgiving of noise (was 2)
    }

    reset() {
        this.state = "HAND_DETECTED";
        this.currentMudra = null;
        this.method = null;
        this.confidence = 0.0;
        this.enterCount = 0;
        this.exitCount = 0;
        this.mismatchCount = 0;
    }

    /**
     * Update FSM with latest detection result
     * @returns {Object} { displayName, displayConf, displayMethod }
     */
    update(handPresent, candidateName, candidateConf, candidateMethod) {
        // NO HAND
        if (!handPresent) {
            this.state = "NO_HAND";
            this.currentMudra = null;
            return { displayName: "No hand detected", displayConf: 0.0, displayMethod: "" };
        }

        // HAND DETECTED
        if (this.state === "NO_HAND") {
            this.state = "HAND_DETECTED";
        }

        if (this.state === "HAND_DETECTED") {
            if (candidateName === null) {
                return { displayName: "Show mudra...", displayConf: 0.0, displayMethod: "" };
            }

            // RULE shortcut → immediate confirmation
            if (candidateMethod === "RULE") {
                this.state = "CONFIRMED_MUDRA";
                this.currentMudra = candidateName;
                this.method = "RULE";
                this.confidence = 1.0;
                return { displayName: this.currentMudra, displayConf: this.confidence, displayMethod: this.method };
            }

            // ML candidate → debounce
            this.state = "ENTERING_MUDRA";
            this.currentMudra = candidateName;
            this.method = candidateMethod;
            this.confidence = candidateConf;
            this.enterCount = 1;
            return { displayName: "Detecting...", displayConf: 0.0, displayMethod: "" };
        }

        // ENTERING
        if (this.state === "ENTERING_MUDRA") {
            if (candidateName === this.currentMudra) {
                this.enterCount++;
                this.mismatchCount = 0;
                this.confidence = candidateConf;

                // Dynamic Threshold for confused pairs
                // If we are prone to confusing Chakra <-> Shivalinga, require more frames
                let requiredFrames = this.ENTER_THRESHOLD;
                if (this._isConfusingPair(this.currentMudra, candidateName)) {
                    requiredFrames = 8; // Require 8 frames (~0.5s) to confirm if confused
                }

                if (this.enterCount >= requiredFrames) {
                    this.state = "CONFIRMED_MUDRA";
                    return { displayName: this.currentMudra, displayConf: this.confidence, displayMethod: this.method };
                } else {
                    return { displayName: "Detecting...", displayConf: 0.0, displayMethod: "" };
                }
            } else {
                this.mismatchCount++;
                if (this.mismatchCount > this.MAX_MISMATCH) {
                    this.reset();
                    return { displayName: "Stabilizing...", displayConf: 0.0, displayMethod: "" };
                }
                return { displayName: "Detecting...", displayConf: 0.0, displayMethod: "" };
            }
        }

        // CONFIRMED
        if (this.state === "CONFIRMED_MUDRA") {
            if (candidateName === this.currentMudra) {
                this.exitCount = 0;
                if (this.method !== "RULE") {
                    this.confidence = candidateConf;
                }
                return { displayName: this.currentMudra, displayConf: this.confidence, displayMethod: this.method };
            } else {
                // If the new candidate is a KNOWN CONFUSION, ignore it longer
                // e.g. if we are in Chakra, and see Shivalinga, assume it's noise unless persistent
                if (this._isConfusingPair(this.currentMudra, candidateName)) {
                    // Increase exist threshold effectively by ignoring mismatches for longer
                    // We handled this by ensuring EXIT_THRESHOLD is respected.
                }

                this.state = "EXITING_MUDRA";
                this.exitCount = 1;
                return { displayName: this.currentMudra, displayConf: this.confidence, displayMethod: this.method };
            }
        }

        // EXITING
        if (this.state === "EXITING_MUDRA") {
            if (candidateName === this.currentMudra) {
                this.state = "CONFIRMED_MUDRA";
                this.exitCount = 0;
                return { displayName: this.currentMudra, displayConf: this.confidence, displayMethod: this.method };
            } else {
                this.exitCount++;

                // STICKY LOGIC: If the potential new mudra is a confusion pair, be extra sticky
                let exitThresh = this.EXIT_THRESHOLD;
                // We don't know the 'new' mudra here easily without state, but generally stickiness helps.

                if (this.exitCount >= exitThresh) {
                    this.reset();
                    return { displayName: "Show mudra...", displayConf: 0.0, displayMethod: "" };
                } else {
                    return { displayName: this.currentMudra, displayConf: this.confidence, displayMethod: this.method };
                }
            }
        }

        return { displayName: "Show mudra...", displayConf: 0.0, displayMethod: "" };
    }

    _isConfusingPair(a, b) {
        if (!a || !b) return false;
        const pairs = [
            ['Chakra', 'Shivalinga'],
            ['Anjali', 'Namaskaram'] // Example
        ];
        return pairs.some(p => (p.includes(a) && p.includes(b)));
    }
}

// ============================================================
// UTILITY FUNCTIONS
// ============================================================

function getDistance(a, b) {
    return Math.hypot(a.x - b.x, a.y - b.y);
}

function getScaleRef(landmarks) {
    return getDistance(landmarks[0], landmarks[9]) + 1e-6;
}

function normDistLazy(landmarks, i, j, scaleRef) {
    return getDistance(landmarks[i], landmarks[j]) / scaleRef;
}

function getAngle(a, b, c, landmarks) {
    try {
        const ax = landmarks[a].x, ay = landmarks[a].y;
        const bx = landmarks[b].x, by = landmarks[b].y;
        const cx = landmarks[c].x, cy = landmarks[c].y;

        const v1 = { x: ax - bx, y: ay - by };
        const v2 = { x: cx - bx, y: cy - by };

        const dot = v1.x * v2.x + v1.y * v2.y;
        const mag1 = Math.hypot(v1.x, v1.y);
        const mag2 = Math.hypot(v2.x, v2.y);

        if (mag1 === 0 || mag2 === 0) return 180;

        const cosA = Math.max(-1, Math.min(1, dot / (mag1 * mag2)));
        return Math.acos(cosA) * (180 / Math.PI);
    } catch (e) {
        return 180;
    }
}

function angleBetween(v1, v2) {
    const mag1 = Math.hypot(v1.x, v1.y);
    const mag2 = Math.hypot(v2.x, v2.y);

    if (mag1 === 0 || mag2 === 0) return null;

    const dot = v1.x * v2.x + v1.y * v2.y;
    const cosA = Math.max(-1, Math.min(1, dot / (mag1 * mag2)));
    return Math.acos(cosA) * (180 / Math.PI);
}

function isFingerStraight(landmarks, mcp, pip, tip, threshold = 0.9) {
    const scaleRef = getScaleRef(landmarks);
    const mcpPip = normDistLazy(landmarks, mcp, pip, scaleRef);
    const pipTip = normDistLazy(landmarks, pip, tip, scaleRef);
    const mcpTip = normDistLazy(landmarks, mcp, tip, scaleRef);

    const total = mcpPip + pipTip;
    if (total === 0) return false;

    const straightnessRatio = mcpTip / total;
    return straightnessRatio > threshold;
}

function isHandSteady(landmarks, prevLandmarks, threshold = 0.02) {
    if (!prevLandmarks) return false;

    const keyIndices = [0, 4, 8, 12, 16, 20];

    for (const idx of keyIndices) {
        const curr = landmarks[idx];
        const prev = prevLandmarks[idx];

        const distance = Math.hypot(
            curr.x - prev.x,
            curr.y - prev.y,
            (curr.z || 0) - (prev.z || 0)
        );

        if (distance > threshold) return false;
    }

    return true;
}

// ============================================================
// RULE-BASED MUDRA DETECTION FUNCTIONS
// ============================================================

function isPatakaMudra(landmarks, scaleRef) {
    const indexStraight = isFingerStraight(landmarks, 5, 6, 8, 0.97);
    const middleStraight = isFingerStraight(landmarks, 9, 10, 12, 0.97);
    const ringStraight = isFingerStraight(landmarks, 13, 14, 16, 0.97);
    const pinkyStraight = isFingerStraight(landmarks, 17, 18, 20, 0.97);

    if (!(indexStraight && middleStraight && ringStraight && pinkyStraight)) return false;

    const index_pinky_nd = normDistLazy(landmarks, 8, 20, scaleRef);

    if (index_pinky_nd > 0.6) return false;

    const thumbIndexNd = normDistLazy(landmarks, 4, 5, scaleRef);
    const mcpNd = normDistLazy(landmarks, 5, 9, scaleRef);
    const thumbTucked = thumbIndexNd < (mcpNd * 1.5);

    return thumbTucked;
}

function isTripatakaMudra(landmarks, scale) {
    if (!isFingerStraight(landmarks, 5, 6, 8, 0.94)) return false;
    if (!isFingerStraight(landmarks, 9, 10, 12, 0.94)) return false;
    if (!isFingerStraight(landmarks, 17, 18, 20, 0.94)) return false;

    if (isFingerStraight(landmarks, 13, 14, 16, 0.88)) return false;

    const index_pinky_nd = normDistLazy(landmarks, 8, 20, scaleRef);

    if (index_pinky_nd > 0.6) return false;

    if (normDistLazy(landmarks, 4, 16, scale) < scale * 0.9) return false;
    if (normDistLazy(landmarks, 4, 5, scale) > scale * 1.6) return false;

    return true;
}

function isMusthiMudra(landmarks, scaleRef) {
    const indexBent = !isFingerStraight(landmarks, 5, 6, 8, 0.8);
    const middleBent = !isFingerStraight(landmarks, 9, 10, 12, 0.8);
    const ringBent = !isFingerStraight(landmarks, 13, 14, 16, 0.8);
    const pinkyBent = !isFingerStraight(landmarks, 17, 18, 20, 0.8);

    if (!(indexBent && middleBent && ringBent && pinkyBent)) return false;

    const thumbIndexPipNd = normDistLazy(landmarks, 4, 6, scaleRef);
    const thumbMiddlePipNd = normDistLazy(landmarks, 4, 10, scaleRef);
    const mcpNd = normDistLazy(landmarks, 5, 9, scaleRef);
    const touchThreshold = mcpNd * 1.5;

    return (thumbIndexPipNd < touchThreshold) || (thumbMiddlePipNd < touchThreshold);
}

function isSuchiMudra(landmarks, scaleRef) {
    const pip = getAngle(5, 6, 7, landmarks);
    const dip = getAngle(6, 7, 8, landmarks);
    if (!(pip > 170 && dip > 170)) return false;

    const wrist = landmarks[0];
    const tip = landmarks[8];
    const mcp = landmarks[5];
    if (Math.hypot(tip.x - wrist.x, tip.y - wrist.y) <=
        Math.hypot(mcp.x - wrist.x, mcp.y - wrist.y)) return false;

    if (isFingerStraight(landmarks, 9, 10, 12, 0.75)) return false;
    if (isFingerStraight(landmarks, 13, 14, 16, 0.75)) return false;
    if (isFingerStraight(landmarks, 17, 18, 20, 0.75)) return false;

    const ref = normDistLazy(landmarks, 5, 9, scaleRef);
    if (ref === 0) return false;

    const tMid1 = normDistLazy(landmarks, 4, 10, scaleRef);
    const tMid2 = normDistLazy(landmarks, 4, 11, scaleRef);
    const tRing1 = normDistLazy(landmarks, 4, 14, scaleRef);
    const tRing2 = normDistLazy(landmarks, 4, 15, scaleRef);
    const touchThresh = ref * 1.6;

    const touchesMidRing = (tMid1 < touchThresh || tMid2 < touchThresh ||
        tRing1 < touchThresh || tRing2 < touchThresh);

    if (!touchesMidRing) return false;
    if (normDistLazy(landmarks, 4, 8, scaleRef) < ref * 2.0) return false;

    return true;
}

function isAralaMudra(landmarks, scaleRef) {
    const indexStraight = isFingerStraight(landmarks, 5, 6, 8, 0.85);
    if (indexStraight) return false;

    const middleStraight = isFingerStraight(landmarks, 9, 10, 12, 0.97);
    const ringStraight = isFingerStraight(landmarks, 13, 14, 16, 0.97);
    const pinkyStraight = isFingerStraight(landmarks, 17, 18, 20, 0.97);

    if (!(middleStraight && ringStraight && pinkyStraight)) return false;

    const dMr = normDistLazy(landmarks, 12, 16, scaleRef);
    const dRp = normDistLazy(landmarks, 16, 20, scaleRef);
    const dMp = normDistLazy(landmarks, 12, 20, scaleRef);
    const closeThresh = scaleRef * 1.2;

    if (!(dMr < closeThresh && dRp < closeThresh && dMp < closeThresh)) return false;

    const thumbStraight = isFingerStraight(landmarks, 2, 3, 4, 0.93);
    if (!thumbStraight) return false;

    return true;
}

function isHamsasyaMudra(landmarks, scaleRef) {
    const dThumbIndexTip = normDistLazy(landmarks, 4, 8, scaleRef);
    if (dThumbIndexTip > 0.28 * scaleRef) return false;

    const indexIsStraight = isFingerStraight(landmarks, 5, 6, 8, 0.94);
    if (indexIsStraight) return false;

    const middleStraight = isFingerStraight(landmarks, 9, 10, 12, 0.93);
    const ringStraight = isFingerStraight(landmarks, 13, 14, 16, 0.92);
    const pinkyStraight = isFingerStraight(landmarks, 17, 18, 20, 0.90);

    if (!(middleStraight && ringStraight && pinkyStraight)) return false;

    return true;
}

function isShikharamMudra(landmarks, scaleRef) {
    const indexBent = !isFingerStraight(landmarks, 5, 6, 8, 0.8);
    const middleBent = !isFingerStraight(landmarks, 9, 10, 12, 0.8);
    const ringBent = !isFingerStraight(landmarks, 13, 14, 16, 0.8);
    const pinkyBent = !isFingerStraight(landmarks, 17, 18, 20, 0.8);

    if (!(indexBent && middleBent && ringBent && pinkyBent)) return false;

    const thumbStraight = isFingerStraight(landmarks, 2, 3, 4, 0.85);
    if (!thumbStraight) return false;

    const thumbIndexNd = normDistLazy(landmarks, 4, 6, scaleRef);
    const thumbMiddleNd = normDistLazy(landmarks, 4, 10, scaleRef);
    const mcpNd = normDistLazy(landmarks, 5, 9, scaleRef);
    const touchThreshold = mcpNd * 1.5;
    const thumbIsTucked = (thumbIndexNd < touchThreshold || thumbMiddleNd < touchThreshold);

    return !thumbIsTucked;
}

function isKartariMukhamMudra(landmarks, scaleRef) {
    const indexStraight = isFingerStraight(landmarks, 5, 6, 8);
    const middleStraight = isFingerStraight(landmarks, 9, 10, 12);
    if (!(indexStraight && middleStraight)) return false;

    const indexTipWristNd = normDistLazy(landmarks, 8, 0, scaleRef);
    const indexMcpWristNd = normDistLazy(landmarks, 5, 0, scaleRef);
    const middleTipWristNd = normDistLazy(landmarks, 12, 0, scaleRef);
    const middleMcpWristNd = normDistLazy(landmarks, 9, 0, scaleRef);

    if (!(indexTipWristNd > indexMcpWristNd && middleTipWristNd > middleMcpWristNd)) return false;

    const ringBent = !isFingerStraight(landmarks, 13, 14, 16, 0.8);
    const pinkyBent = !isFingerStraight(landmarks, 17, 18, 20, 0.8);
    if (!(ringBent && pinkyBent)) return false;

    const thumbRingPipNd = normDistLazy(landmarks, 4, 14, scaleRef);
    const thumbRingTipNd = normDistLazy(landmarks, 4, 16, scaleRef);
    const mcpNd = normDistLazy(landmarks, 5, 9, scaleRef);
    const touchThreshold = mcpNd * 2.0;

    return (thumbRingPipNd < touchThreshold) || (thumbRingTipNd < touchThreshold);
}

function isSarpashirshaMudra(landmarks, scaleRef) {
    const loose = 0.75;  // relaxed straightness

    if (!(
        isFingerStraight(landmarks, 5, 6, 8, loose) &&
        isFingerStraight(landmarks, 9, 10, 12, loose) &&
        isFingerStraight(landmarks, 13, 14, 16, loose) &&
        isFingerStraight(landmarks, 17, 18, 20, loose)
    )) return false;

    const tipDist = normDistLazy(landmarks, 8, 20, scaleRef);
    const mcpDist = normDistLazy(landmarks, 5, 17, scaleRef);

    // Relaxed convergence
    if (tipDist > mcpDist * 0.90) return false;

    // Thumb loosely tucked
    const thumbIndexNd = normDistLazy(landmarks, 4, 5, scaleRef);
    const mcpRefNd = normDistLazy(landmarks, 5, 9, scaleRef);

    if (thumbIndexNd > mcpRefNd * 2.0) return false;

    return true;
}

function isChaturaMudra(landmarks, scaleRef) {
    if (!(isFingerStraight(landmarks, 5, 6, 8, 0.85) &&
        isFingerStraight(landmarks, 9, 10, 12, 0.85) &&
        isFingerStraight(landmarks, 13, 14, 16, 0.85) &&
        isFingerStraight(landmarks, 17, 18, 20, 0.85))) return false;

    const indexTip = landmarks[8];
    const middleTip = landmarks[12];
    const ringTip = landmarks[16];

    const dIm = Math.hypot(indexTip.x - middleTip.x, indexTip.y - middleTip.y);
    const dMr = Math.hypot(middleTip.x - ringTip.x, middleTip.y - ringTip.y);

    if (dIm > scaleRef * 0.30 || dMr > scaleRef * 0.30) return false;

    const wrist = landmarks[0];
    const middleMcp = [landmarks[9].x, landmarks[9].y];
    const palmCenter = [(wrist.x + middleMcp[0]) / 2, (wrist.y + middleMcp[1]) / 2];

    const thumbTip = [landmarks[4].x, landmarks[4].y];
    const indexMcp = [landmarks[5].x, landmarks[5].y];
    const pinkyMcp = [landmarks[17].x, landmarks[17].y];

    const dThumbPalm = Math.hypot(thumbTip[0] - palmCenter[0], thumbTip[1] - palmCenter[1]);
    const dThumbIndex = Math.hypot(thumbTip[0] - landmarks[8].x, thumbTip[1] - landmarks[8].y);
    const dThumbMiddle = Math.hypot(thumbTip[0] - landmarks[12].x, thumbTip[1] - landmarks[12].y);
    const dThumbRing = Math.hypot(thumbTip[0] - landmarks[16].x, thumbTip[1] - landmarks[16].y);
    const dThumbPinky = Math.hypot(thumbTip[0] - landmarks[20].x, thumbTip[1] - landmarks[20].y);

    const tolerance = scaleRef * 0.01;
    const thumbDepthOk = thumbTip[1] > (middleMcp[1] - tolerance);
    const thumbXInside = Math.min(indexMcp[0], pinkyMcp[0]) < thumbTip[0] &&
        thumbTip[0] < Math.max(indexMcp[0], pinkyMcp[0]);
    const thumbDeepInside = (dThumbPalm < dThumbIndex && dThumbPalm < dThumbMiddle &&
        dThumbPalm < dThumbRing && dThumbPalm < dThumbPinky);

    if (!(thumbDepthOk && thumbDeepInside && thumbXInside)) return false;

    return true;
}

function isTrishulaMudra(landmarks, scaleRef) {
    const indexStraight = isFingerStraight(landmarks, 5, 6, 8);
    const middleStraight = isFingerStraight(landmarks, 9, 10, 12);
    const ringStraight = isFingerStraight(landmarks, 13, 14, 16);
    if (!(indexStraight && middleStraight && ringStraight)) return false;

    const pinkyBent = !isFingerStraight(landmarks, 17, 18, 20, 0.8);
    if (!pinkyBent) return false;

    const thumbPinkyNd = normDistLazy(landmarks, 4, 20, scaleRef);
    const mcpNd = normDistLazy(landmarks, 5, 9, scaleRef);

    return thumbPinkyNd < (mcpNd * 1.0);
}

function isMrigasheersha(landmarks, scaleRef) {
    const pinkyStraight = isFingerStraight(landmarks, 17, 18, 20, 0.85);
    const thumbStraight = isFingerStraight(landmarks, 2, 3, 4, 0.80);

    const indexBent = !isFingerStraight(landmarks, 5, 6, 8, 0.85);
    const middleBent = !isFingerStraight(landmarks, 9, 10, 12, 0.85);
    const ringBent = !isFingerStraight(landmarks, 13, 14, 16, 0.85);

    if (!(pinkyStraight && thumbStraight && indexBent && middleBent && ringBent)) return false;

    const vThumb = { x: landmarks[4].x - landmarks[2].x, y: landmarks[4].y - landmarks[2].y };
    const vHand = { x: landmarks[9].x - landmarks[0].x, y: landmarks[9].y - landmarks[0].y };
    const ang = angleBetween(vThumb, vHand);

    if (ang === null) return false;

    return ang > 35;
}

function isSimhamukha(landmarks, scaleRef) {
    const ref = normDistLazy(landmarks, 5, 9, scaleRef);
    if (ref === 0) return false;

    if (!isFingerStraight(landmarks, 5, 6, 8, 0.90)) return false;
    if (!isFingerStraight(landmarks, 17, 18, 20, 0.90)) return false;

    const dTm = normDistLazy(landmarks, 4, 12, scaleRef);
    const dTr = normDistLazy(landmarks, 4, 16, scaleRef);
    const dMr = normDistLazy(landmarks, 12, 16, scaleRef);
    const clusterThresh = ref * 1.9;

    if (!(dTm < clusterThresh && dTr < clusterThresh && dMr < clusterThresh)) return false;

    const midStraight = isFingerStraight(landmarks, 9, 10, 12, 0.92);
    const ringStraight = isFingerStraight(landmarks, 13, 14, 16, 0.92);

    if (midStraight && landmarks[12].y < landmarks[9].y) return false;
    if (ringStraight && landmarks[16].y < landmarks[13].y) return false;

    return true;
}

function isHamsapakashaMudra(landmarks, scaleRef) {
    // Pinky and thumb must be straight
    const pinkyStraight = isFingerStraight(landmarks, 17, 18, 20, 0.85);
    const thumbStraight = isFingerStraight(landmarks, 2, 3, 4, 0.80);

    // Index, middle, ring must be bent
    const indexBent = !isFingerStraight(landmarks, 5, 6, 8, 0.85);
    const middleBent = !isFingerStraight(landmarks, 9, 10, 12, 0.85);
    const ringBent = !isFingerStraight(landmarks, 13, 14, 16, 0.85);

    if (!(pinkyStraight && thumbStraight && indexBent && middleBent && ringBent)) return false;

    // Thumb must touch index MCP (start of index finger)
    const thumbIndexMcpNd = normDistLazy(landmarks, 4, 5, scaleRef);
    if (thumbIndexMcpNd > 0.45) return false;

    return true;
}

function isArdhapataka(landmarks, scaleRef) {
    const indexStraight = isFingerStraight(landmarks, 5, 6, 8);
    const middleStraight = isFingerStraight(landmarks, 9, 10, 12);
    if (!(indexStraight && middleStraight)) return false;

    const ringBent = !isFingerStraight(landmarks, 13, 14, 16, 0.8);
    const pinkyBent = !isFingerStraight(landmarks, 17, 18, 20, 0.8);
    if (!(ringBent && pinkyBent)) return false;

    const thumbIndexNd = normDistLazy(landmarks, 4, 5, scaleRef);
    const mcpNd = normDistLazy(landmarks, 5, 9, scaleRef);

    return thumbIndexNd < (mcpNd * 1.5);
}

/**
 * Detects Mayura Mudra:
 * - Ring finger tip touches Thumb tip.
 * - Index, Middle, and Pinky fingers are extended straight.
 */
/**
 * Detects Mayura Mudra:
 * - Ring finger tip touches Thumb tip.
 * - Index, Middle, and Pinky fingers are straight.
 * - Index and Pinky fingers are NOT spread apart (held relatively close).
 */
function isMayuraMudra(landmarks, scaleRef) {
    // 1. Check Extended Fingers (Index, Middle, Pinky)
    // Threshold 0.85 allows for slight natural curvature.
    if (!isFingerStraight(landmarks, 5, 6, 8, 0.85)) return false;
    if (!isFingerStraight(landmarks, 9, 10, 12, 0.85)) return false;
    if (!isFingerStraight(landmarks, 17, 18, 20, 0.85)) return false;

    // 2. Check Connection: Ring Tip (16) touching Thumb Tip (4)
    // Threshold 0.1 ensures they are actually touching.
    const thumbRingDist = normDistLazy(landmarks, 4, 16, scaleRef);
    if (thumbRingDist > 0.1) return false;

    // 3. Check Ring Finger Bend
    // The ring finger must be bent to touch the thumb.
    if (isFingerStraight(landmarks, 13, 14, 16, 0.90)) return false;

    // 4. Spread Check (Index vs Pinky)
    // We calculate the distance between Index Tip (8) and Pinky Tip (20).
    // If this distance is too large, the fingers are splayed.
    const indexToPinkyDist = normDistLazy(landmarks, 8, 20, scaleRef);

    // 0.6 is the threshold. If distance is > 60% of the palm size, 
    // the fingers are likely spread too wide.
    if (indexToPinkyDist > 0.6) return false;

    return true;
}

function isShukaTundam(landmarks, scaleRef) {
    const middleStraight = isFingerStraight(landmarks, 9, 10, 12);
    const pinkyStraight = isFingerStraight(landmarks, 17, 18, 20);

    const indexBent = !isFingerStraight(landmarks, 5, 6, 8, 0.85);
    const ringBent = !isFingerStraight(landmarks, 13, 14, 16, 0.85);

    if (!(middleStraight && pinkyStraight && indexBent && ringBent)) return false;

    const thumbIndexNd = normDistLazy(landmarks, 4, 5, scaleRef);
    const mcpNd = normDistLazy(landmarks, 5, 9, scaleRef);

    return thumbIndexNd < (mcpNd * 1.5);
}

// ============================================================
// RULE-BASED MUDRA REGISTRY
// ============================================================

const RULE_MUDRA_FUNCTIONS = {
    "Musthi Mudra": isMusthiMudra,
    "Suchi Mudra": isSuchiMudra,
    "Shikharam Mudra": isShikharamMudra,
    "Hamsasya Mudra": isHamsasyaMudra,
    "Mayura Mudra": isMayuraMudra,
    "Tripataka Mudra": isTripatakaMudra,
    "Kartari Mukham Mudra": isKartariMukhamMudra,
    "Trishula Mudra": isTrishulaMudra,
    "Mrigasheersha Mudra": isMrigasheersha,
    "Hamsapaksha Mudra": isHamsapakashaMudra,
    "Ardhapataka Mudra": isArdhapataka,
    "Shuka Tundam Mudra": isShukaTundam,
    "Arala Mudra": isAralaMudra,
    "Sarpashirsha Mudra": isSarpashirshaMudra,
    "Pataka Mudra": isPatakaMudra,
    "Chatura Mudra": isChaturaMudra
};

// ============================================================
// HYBRID DETECTION PIPELINE
// ============================================================

const ML_CONF_THRESHOLD = 0.55;

/**
 * Hybrid decision logic with Weighted Voting and Temporal Smoothing
 * @param {Array} landmarks - MediaPipe hand landmarks
 * @param {string} handednessLabel - "Left" or "Right"
 * @param {Array} prevLandmarks - Previous frame landmarks
 * @returns {Object} { mudraName, confidence, method }
 */
async function detectMudraHybrid(landmarks, handednessLabel, prevLandmarks) {
    const scaleRef = getScaleRef(landmarks);

    // Candidates map: { mudraName: score }
    const candidates = {};

    // Helper to add score with Normalization
    const addScore = (name, score, source) => {
        if (!name) return;

        // NORMALIZE NAME: Ensure "Pathaka" (ML) and "Pataka Mudra" (Rules) become "Pataka"
        let normalizedName = name;
        if (window.MudraConstants && window.MudraConstants.normalizeMudraName) {
            normalizedName = window.MudraConstants.normalizeMudraName(name);
        } else {
            // Fallback if constants not loaded
            normalizedName = name.replace(" Mudra", "");
        }

        if (!candidates[normalizedName]) candidates[normalizedName] = { score: 0, sources: [] };
        candidates[normalizedName].score += score;
        candidates[normalizedName].sources.push(source);
    };

    // 1. RULE-BASED VOTING
    // Rules provide strong evidence but can be brittle.
    // We assign 1.2 score (previously 0.8) so they override ML (max 1.0) if a rule matches.
    for (const [mudraName, checkFunc] of Object.entries(RULE_MUDRA_FUNCTIONS)) {
        try {
            if (checkFunc(landmarks, scaleRef)) {
                addScore(mudraName, 1.2, "RULE");
            }
        } catch (e) {
            continue;
        }
    }

    // 2. ML VOTING
    // Always run ML (removed isHandSteady block) to support movement
    try {
        if (window.MudraInference && window.MudraInference.isInitialized()) {
            const result = await window.MudraInference.detectSingleHandML(landmarks);

            if (result.className && result.confidence > 0.3) {
                // Add ML score (0.0 to 1.0)
                addScore(result.className, result.confidence, "ML");
            }
        }
    } catch (e) {
        console.error("ML prediction error:", e);
    }

    // 3. WINNER SELECTION
    let winner = null;
    let maxScore = 0;

    for (const [name, data] of Object.entries(candidates)) {
        if (data.score > maxScore) {
            maxScore = data.score;
            winner = name;
        }
    }

    // 4. CONFIDENCE THRESHOLD
    const HYBRID_THRESHOLD = 0.55;

    if (winner && maxScore > HYBRID_THRESHOLD) {
        // Normalize confidence slightly for display (cap at 1.0)
        const finalConf = Math.min(1.0, maxScore);
        const method = candidates[winner].sources.includes("RULE") ? "HYBRID" : "ML";

        return { mudraName: winner, confidence: finalConf, method: method };
    }

    return { mudraName: "Unknown", confidence: 0.0, method: "ML" };
}

// Export for use in other modules
window.MudraDetection = {
    MudraFSM,
    detectMudraHybrid,
    isHandSteady,
    getScaleRef,
    RULE_MUDRA_FUNCTIONS,
    ML_CONF_THRESHOLD
};
