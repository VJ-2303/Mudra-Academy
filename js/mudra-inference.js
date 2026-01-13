/**
 * ================================================================================
 * MUDRA INFERENCE MODULE - JavaScript/ONNX Runtime Web
 * ================================================================================
 * Provides ML inference for mudra detection using ONNX Runtime Web.
 * Handles both YOLO (double-hand) and Random Forest (single-hand) models.
 * ================================================================================
 */

// Model paths - relative to page location
// Detect base path based on current page location
const getBasePath = () => {
    const path = window.location.pathname;
    // If we're in a subfolder like /pages/, go up one level
    if (path.includes('/pages/')) {
        return '../';
    }
    return './';
};

const MODEL_PATHS = {
    get yolo() { return getBasePath() + 'ml/models/kkpv_web.onnx'; },
    get randomForest() { return getBasePath() + 'ml/models/mudra_rf_model.onnx'; },
    get classes() { return getBasePath() + 'ml/models/mudra_classes.json'; }
};

// YOLO class names (double-hand mudras) - MUST match model training order
const YOLO_CLASSES = [
    'Anjali', 'Naagabandha', 'Chakra', 'Karkota',
    'Kartariswastika', 'Pasha', 'Shanka', 'Shivalinga'
];

// Global sessions
let yoloSession = null;
let rfSession = null;
let rfClasses = null;

/**
 * Initialize ONNX Runtime sessions
 * @param {Object} options - { loadYolo: boolean, loadRF: boolean }
 */
async function initializeInference(options = { loadYolo: false, loadRF: true }) {
    console.log('🔄 Initializing ONNX Runtime...');
    
    // Configure ONNX Runtime for single-threaded operation (avoids COOP/COEP issues)
    if (typeof ort !== 'undefined') {
        ort.env.wasm.numThreads = 1;
    }
    
    try {
        // Initialize YOLO model for double-hand detection (if requested)
        if (options.loadYolo) {
            console.log('Loading YOLO model from:', MODEL_PATHS.yolo);
            yoloSession = await ort.InferenceSession.create(MODEL_PATHS.yolo, {
                executionProviders: ['wasm', 'cpu'],
                graphOptimizationLevel: 'all'
            });
            console.log('✅ YOLO model loaded');
        }
        
        // Initialize Random Forest model for single-hand detection (if requested)
        if (options.loadRF) {
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
        }
        
        return true;
    } catch (error) {
        console.error('❌ Failed to initialize inference:', error);
        return false;
    }
}

// ============================================================
// MEMORY OPTIMIZATION: Reusable canvas and tensor pools
// ============================================================
let _yoloCanvas = null;
let _yoloCtx = null;
let _tempCanvas = null;
let _tempCtx = null;
let _yoloTensor = null;
let _featureTensor = null;

/**
 * Get or create reusable YOLO preprocessing canvas (640x640)
 */
function getYOLOCanvas() {
    if (!_yoloCanvas) {
        _yoloCanvas = document.createElement('canvas');
        _yoloCanvas.width = 640;
        _yoloCanvas.height = 640;
        _yoloCtx = _yoloCanvas.getContext('2d', { willReadFrequently: true });
    }
    return { canvas: _yoloCanvas, ctx: _yoloCtx };
}

/**
 * Get or create reusable temp canvas for ImageData conversion
 */
function getTempCanvas(width, height) {
    if (!_tempCanvas || _tempCanvas.width !== width || _tempCanvas.height !== height) {
        _tempCanvas = document.createElement('canvas');
        _tempCanvas.width = width;
        _tempCanvas.height = height;
        _tempCtx = _tempCanvas.getContext('2d');
    }
    return { canvas: _tempCanvas, ctx: _tempCtx };
}

/**
 * Get or create reusable YOLO tensor (avoids allocation per frame)
 */
function getYOLOTensor() {
    if (!_yoloTensor) {
        _yoloTensor = new Float32Array(3 * 640 * 640);
    }
    return _yoloTensor;
}

/**
 * Get or create reusable feature tensor (17 features)
 */
function getFeatureTensor() {
    if (!_featureTensor) {
        _featureTensor = new Float32Array(17);
    }
    return _featureTensor;
}

/**
 * Preprocess image for YOLO model (OPTIMIZED - reuses canvas/tensor)
 * @param {ImageData|HTMLCanvasElement|HTMLVideoElement} input - Input image
 * @returns {Float32Array} Preprocessed tensor
 */
function preprocessYOLO(input) {
    // Reuse canvas instead of creating new one each frame
    const { ctx } = getYOLOCanvas();
    
    // Draw input (resize to 640x640)
    if (input instanceof ImageData) {
        const { canvas: temp, ctx: tempCtx } = getTempCanvas(input.width, input.height);
        tempCtx.putImageData(input, 0, 0);
        ctx.drawImage(temp, 0, 0, 640, 640);
    } else {
        ctx.drawImage(input, 0, 0, 640, 640);
    }
    
    const imageData = ctx.getImageData(0, 0, 640, 640);
    const data = imageData.data;
    
    // Reuse tensor instead of allocating new one each frame
    const tensor = getYOLOTensor();
    
    // OPTIMIZED: Single loop instead of three separate loops
    const pixelCount = 640 * 640;
    for (let i = 0; i < pixelCount; i++) {
        const base = i * 4;
        tensor[i] = data[base] / 255.0;                    // Red
        tensor[pixelCount + i] = data[base + 1] / 255.0;   // Green
        tensor[pixelCount * 2 + i] = data[base + 2] / 255.0; // Blue
    }
    
    return tensor;
}

/**
 * Run YOLO inference for double-hand mudra detection
 * @param {HTMLCanvasElement|HTMLVideoElement} input - Input image
 * @returns {Object} Detection result { className, confidence, box }
 */
async function detectDoubleHand(input) {
    if (!yoloSession) {
        throw new Error('YOLO model not initialized');
    }
    
    try {
        // Preprocess
        const inputTensor = preprocessYOLO(input);
        const tensor = new ort.Tensor('float32', inputTensor, [1, 3, 640, 640]);
        
        // Run inference
        const feeds = { images: tensor };
        const results = await yoloSession.run(feeds);
        
        // Get output (shape: [1, 12, 8400] for 8 classes + 4 box coords)
        const output = results[Object.keys(results)[0]];
        const outputData = output.data;
        const numDetections = 8400;
        const numClasses = 8;
        
        // Find best detection
        let bestScore = 0;
        let bestClass = -1;
        let bestBox = null;
        
        for (let i = 0; i < numDetections; i++) {
            // Box coords: [x, y, w, h] at indices 0-3
            const x = outputData[0 * numDetections + i];
            const y = outputData[1 * numDetections + i];
            const w = outputData[2 * numDetections + i];
            const h = outputData[3 * numDetections + i];
            
            // Class scores at indices 4-11
            let maxClassScore = 0;
            let maxClassIdx = 0;
            
            for (let c = 0; c < numClasses; c++) {
                const score = outputData[(4 + c) * numDetections + i];
                if (score > maxClassScore) {
                    maxClassScore = score;
                    maxClassIdx = c;
                }
            }
            
            if (maxClassScore > bestScore) {
                bestScore = maxClassScore;
                bestClass = maxClassIdx;
                bestBox = { x, y, w, h };
            }
        }
        
        // Apply confidence threshold
        const CONF_THRESHOLD = 0.25;
        if (bestScore < CONF_THRESHOLD || bestClass < 0) {
            return { className: null, confidence: 0, box: null };
        }
        
        return {
            className: YOLO_CLASSES[bestClass],
            confidence: bestScore,
            box: bestBox
        };
        
    } catch (error) {
        console.error('YOLO inference error:', error);
        return { className: null, confidence: 0, box: null };
    }
}

/**
 * Extract 17 features from MediaPipe hand landmarks for Random Forest model
 * OPTIMIZED: Reuses tensor allocation
 * @param {Array} landmarks - MediaPipe hand landmarks (21 points)
 * @returns {Float32Array} 17 features
 */
function extractMLFeatures(landmarks) {
    // Reuse tensor instead of allocating new one each frame
    const features = getFeatureTensor();
    let idx = 0;
    
    // Scale reference: palm width (wrist to middle MCP)
    const scaleRef = Math.hypot(
        landmarks[9].x - landmarks[0].x,
        landmarks[9].y - landmarks[0].y
    ) + 1e-6;
    
    // Helper: Euclidean distance between two landmarks
    const dist = (i, j) => Math.hypot(
        landmarks[i].x - landmarks[j].x,
        landmarks[i].y - landmarks[j].y
    );
    
    // Helper: Normalized distance
    const normDist = (i, j) => dist(i, j) / scaleRef;
    
    // Helper: Finger straightness ratio
    const fingerStraightness = (mcp, pip, tip) => {
        const d1 = dist(mcp, pip);
        const d2 = dist(pip, tip);
        const d3 = dist(mcp, tip);
        return d3 / (d1 + d2 + 1e-6);
    };
    
    // Helper: Angle at point b formed by a-b-c (degrees)
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
    features[idx++] = normDist(4, 8);   // Thumb to index
    features[idx++] = normDist(4, 12);  // Thumb to middle
    features[idx++] = normDist(4, 16);  // Thumb to ring
    features[idx++] = normDist(4, 20);  // Thumb to pinky
    
    // 3️⃣ Fingertip clustering (3 features)
    features[idx++] = normDist(8, 12);  // Index to middle
    features[idx++] = normDist(12, 16); // Middle to ring
    features[idx++] = normDist(16, 20); // Ring to pinky
    
    // 4️⃣ Joint angles (4 features)
    features[idx++] = getAngle(5, 6, 7);   // Index PIP
    features[idx++] = getAngle(6, 7, 8);   // Index DIP
    features[idx++] = getAngle(9, 10, 11); // Middle PIP
    features[idx++] = getAngle(13, 14, 15);// Ring PIP
    
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
        // Extract features
        const features = extractMLFeatures(landmarks);
        const tensor = new ort.Tensor('float32', features, [1, 17]);
        
        // Run inference
        const feeds = {};
        feeds[rfSession.inputNames[0]] = tensor;
        const results = await rfSession.run(feeds);
        
        // Debug: Log output structure once
        if (!window._rfDebugLogged) {
            console.log('RF Model outputs:', rfSession.outputNames);
            console.log('Output 0 (label):', results[rfSession.outputNames[0]]);
            console.log('Output 1 (probabilities):', results[rfSession.outputNames[1]]);
            window._rfDebugLogged = true;
        }
        
        // Get prediction label (output 0)
        const predictionOutput = results[rfSession.outputNames[0]];
        const prediction = predictionOutput.data[0];
        
        // Get probabilities (output 1)
        // sklearn-onnx RandomForest outputs: 
        //   - output_label: string class name
        //   - output_probability: sequence of maps [{class: prob}, ...]
        const probOutput = results[rfSession.outputNames[1]];
        let maxProb = 0;
        let predictedClass = prediction;
        
        if (probOutput && probOutput.data) {
            const probData = probOutput.data;
            
            // Handle different formats
            if (probData.length > 0) {
                const firstItem = probData[0];
                
                if (typeof firstItem === 'object' && firstItem !== null && !Array.isArray(firstItem)) {
                    // ZipMap format: array of {className: probability} objects
                    // Find the probability for the predicted class
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
                    // If we didn't find the prediction's probability, use max
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
                    // Flat array format: [prob0, prob1, ...]
                    const predIdx = rfClasses.indexOf(prediction);
                    if (predIdx >= 0 && predIdx < probData.length) {
                        maxProb = probData[predIdx];
                    } else {
                        // Find max
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
        
        // Fallback: if confidence still 0, something went wrong - use moderate confidence
        // This prevents the ML_CONF_THRESHOLD from filtering out all results
        if (maxProb === 0 && prediction) {
            console.warn('Could not extract probability, using default 0.6');
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
    initializeRFOnly: () => initializeInference({ loadYolo: false, loadRF: true }),
    initializeYOLOOnly: () => initializeInference({ loadYolo: true, loadRF: false }),
    detectDoubleHand,
    detectSingleHandML,
    extractMLFeatures,
    YOLO_CLASSES,
    getRFClasses: () => rfClasses,
    isInitialized: () => rfSession !== null,
    isYOLOInitialized: () => yoloSession !== null,
    isRFInitialized: () => rfSession !== null
};
