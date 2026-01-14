/**
 * ================================================================================
 * MUDRA INFERENCE MODULE - JavaScript/ONNX Runtime Web
 * ================================================================================
 * Provides ML inference for single-hand mudra detection using Random Forest.
 * Note: Double-hand YOLO detection is handled in double-hand-detection.html
 * ================================================================================
 */

// Model paths - relative to page location
const getBasePath = () => {
    const path = window.location.pathname;
    if (path.includes('/pages/')) {
        return '../';
    }
    return './';
};

const MODEL_PATHS = {
    get randomForest() { return getBasePath() + 'ml/models/mudra_rf_model_new.onnx'; },
    get classes() { return getBasePath() + 'ml/models/mudra_classes_new.json'; }
};

// Global sessions
let rfSession = null;
let rfClasses = null;

/**
 * Initialize ONNX Runtime session for Random Forest model
 */
async function initializeInference() {
    console.log('🔄 Initializing ONNX Runtime...');

    // Configure ONNX Runtime for single-threaded operation (avoids COOP/COEP issues)
    if (typeof ort !== 'undefined') {
        ort.env.wasm.numThreads = 1;
    }

    try {
        console.log('Loading RF model from:', MODEL_PATHS.randomForest);
        rfSession = await ort.InferenceSession.create(MODEL_PATHS.randomForest, {
            executionProviders: ['wasm', 'cpu'],
            graphOptimizationLevel: 'all'
        });
        console.log('✅ Random Forest model loaded');

        // Load class names
        console.log('Loading classes from:', MODEL_PATHS.classes);
        const response = await fetch(MODEL_PATHS.classes);
        rfClasses = await response.json();
        console.log(`✅ Loaded ${rfClasses.length} RF classes`);

        return true;
    } catch (error) {
        console.error('❌ Failed to initialize inference:', error);
        return false;
    }
}

// ============================================================
// MEMORY OPTIMIZATION: Reusable tensor
// ============================================================
let _featureTensor = null;

function getFeatureTensor() {
    if (!_featureTensor) {
        _featureTensor = new Float32Array(17);
    }
    return _featureTensor;
}

/**
 * Extract 17 features from MediaPipe hand landmarks for Random Forest model
 * @param {Array} landmarks - MediaPipe hand landmarks (21 points)
 * @returns {Float32Array} 17 features
 */
function extractMLFeatures(landmarks) {
    const features = getFeatureTensor();
    let idx = 0;

    // Scale reference: palm width (wrist to middle MCP)
    const scaleRef = Math.hypot(
        landmarks[9].x - landmarks[0].x,
        landmarks[9].y - landmarks[0].y
    ) + 1e-6;

    const dist = (i, j) => Math.hypot(
        landmarks[i].x - landmarks[j].x,
        landmarks[i].y - landmarks[j].y
    );

    const normDist = (i, j) => dist(i, j) / scaleRef;

    const fingerStraightness = (mcp, pip, tip) => {
        const d1 = dist(mcp, pip);
        const d2 = dist(pip, tip);
        const d3 = dist(mcp, tip);
        return d3 / (d1 + d2 + 1e-6);
    };

    const getAngle = (a, b, c) => {
        const v1x = landmarks[a].x - landmarks[b].x;
        const v1y = landmarks[a].y - landmarks[b].y;
        const v2x = landmarks[c].x - landmarks[b].x;
        const v2y = landmarks[c].y - landmarks[b].y;

        const dot = v1x * v2x + v1y * v2y;
        const mag1 = Math.hypot(v1x, v1y);
        const mag2 = Math.hypot(v2x, v2y);

        if (mag1 === 0 || mag2 === 0) return 180;

        const cosA = Math.max(-1, Math.min(1, dot / (mag1 * mag2)));
        return Math.acos(cosA) * (180 / Math.PI);
    };

    // 1️⃣ Finger straightness (5 features)
    features[idx++] = fingerStraightness(2, 3, 4);    // Thumb
    features[idx++] = fingerStraightness(5, 6, 8);    // Index
    features[idx++] = fingerStraightness(9, 10, 12);  // Middle
    features[idx++] = fingerStraightness(13, 14, 16); // Ring
    features[idx++] = fingerStraightness(17, 18, 20); // Pinky

    // 2️⃣ Thumb-to-fingertip distances (4 features)
    features[idx++] = normDist(4, 8);
    features[idx++] = normDist(4, 12);
    features[idx++] = normDist(4, 16);
    features[idx++] = normDist(4, 20);

    // 3️⃣ Fingertip clustering (3 features)
    features[idx++] = normDist(8, 12);
    features[idx++] = normDist(12, 16);
    features[idx++] = normDist(16, 20);

    // 4️⃣ Joint angles (4 features)
    features[idx++] = getAngle(5, 6, 7);
    features[idx++] = getAngle(6, 7, 8);
    features[idx++] = getAngle(9, 10, 11);
    features[idx++] = getAngle(13, 14, 15);

    // 5️⃣ Palm orientation Z-axis (1 feature)
    features[idx++] = Math.abs(landmarks[9].z - landmarks[0].z) / scaleRef;

    return features;
}

/**
 * Run Random Forest inference for single-hand mudra detection
 * @param {Array} landmarks - MediaPipe hand landmarks
 * @returns {Object} Prediction { className, confidence, probabilities }
 */
async function detectSingleHandML(landmarks) {
    if (!rfSession || !rfClasses) {
        throw new Error('Random Forest model not initialized');
    }

    try {
        const features = extractMLFeatures(landmarks);
        const tensor = new ort.Tensor('float32', features, [1, 17]);

        const feeds = {};
        feeds[rfSession.inputNames[0]] = tensor;
        const results = await rfSession.run(feeds);

        // Get prediction label (output 0)
        const predictionOutput = results[rfSession.outputNames[0]];
        const prediction = predictionOutput.data[0];

        // Get probabilities (output 1)
        const probOutput = results[rfSession.outputNames[1]];
        let maxProb = 0;
        let predictedClass = prediction;

        if (probOutput && probOutput.data) {
            const probData = probOutput.data;

            if (probData.length > 0) {
                const firstItem = probData[0];

                if (typeof firstItem === 'object' && firstItem !== null && !Array.isArray(firstItem)) {
                    // ZipMap format
                    for (const entry of probData) {
                        if (entry && typeof entry === 'object') {
                            for (const [className, prob] of Object.entries(entry)) {
                                if (className === prediction) {
                                    maxProb = prob;
                                    break;
                                }
                            }
                        }
                    }
                    if (maxProb === 0) {
                        for (const entry of probData) {
                            if (entry && typeof entry === 'object') {
                                for (const [className, prob] of Object.entries(entry)) {
                                    if (prob > maxProb) {
                                        maxProb = prob;
                                        predictedClass = className;
                                    }
                                }
                            }
                        }
                    }
                } else if (typeof firstItem === 'number') {
                    // Flat array format
                    const predIdx = rfClasses.indexOf(prediction);
                    if (predIdx >= 0 && predIdx < probData.length) {
                        maxProb = probData[predIdx];
                    } else {
                        let maxIdx = 0;
                        for (let i = 0; i < probData.length; i++) {
                            if (probData[i] > maxProb) {
                                maxProb = probData[i];
                                maxIdx = i;
                            }
                        }
                        if (rfClasses && maxIdx < rfClasses.length) {
                            predictedClass = rfClasses[maxIdx];
                        }
                    }
                }
            }
        }

        // Fallback confidence
        if (maxProb === 0 && prediction) {
            maxProb = 0.6;
        }

        return {
            className: predictedClass || prediction,
            confidence: maxProb,
            probabilities: probOutput?.data ? Array.from(probOutput.data) : null
        };

    } catch (error) {
        console.error('RF inference error:', error);
        return { className: null, confidence: 0, probabilities: null };
    }
}

// Export for use in other modules
window.MudraInference = {
    initialize: initializeInference,
    detectSingleHandML,
    extractMLFeatures,
    getRFClasses: () => rfClasses,
    isInitialized: () => rfSession !== null
};

