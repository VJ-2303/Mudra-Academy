# Mudra Academy - Pure JavaScript/Electron Edition

This is a complete offline Bharatanatyam mudra learning platform built with:
- **Electron** for desktop application
- **ONNX Runtime Web** for ML inference
- **MediaPipe Hands** for hand landmark detection

## Features

### Single Hand Mudras (28 mudras)
- Hybrid detection: Rule-based geometric checks + Random Forest ML
- Finite State Machine (FSM) for flicker-free output
- Real-time hand landmark visualization

### Double Hand Mudras (8 mudras)  
- YOLOv8 object detection
- Bounding box visualization
- High accuracy detection

### Ghost Practice Mode
- Visual overlay for alignment practice
- Voice control support

## Running the Application

### Option 1: Electron Desktop App
```bash
npm install
npm start
```

### Option 2: Local Web Server (for development)
```bash
python3 -m http.server 8080
# Open http://localhost:8080 in browser
```

### Option 3: Direct File Opening
Open `index.html` directly in a modern browser (Chrome/Edge recommended)

## Technical Architecture

### Models (all in `ml/models/`)
- `kkpv_web.onnx` (12MB) - YOLOv8 for double-hand detection
- `mudra_rf_model.onnx` (60MB) - Random Forest for single-hand ML
- `mudra_classes.json` - Class labels for RF model

### JavaScript Modules
- `js/mudra-inference.js` - ONNX Runtime Web inference wrapper
- `js/mudra-detection.js` - FSM + rule-based detection logic

### Pages
- `pages/live-detection-offline.html` - Single hand detection (offline)
- `pages/double-hand-detection-offline.html` - Double hand detection (offline)
- `pages/detection-mode.html` - Mode selection

## Model Accuracy

The ONNX models maintain the same accuracy as the original Python models:
- **Random Forest**: Exact match (verified with random test inputs)
- **YOLOv8**: ~7% confidence difference (class predictions identical)

## Dependencies

### Runtime (loaded via CDN in HTML)
- ONNX Runtime Web 1.17.0
- MediaPipe Hands 0.4
- MediaPipe Camera Utils 0.3

### Development
- Electron 28.0.0

## Browser Compatibility
- Chrome 88+
- Edge 88+
- Firefox 78+ (limited WASM performance)
- Safari 14+ (WebAssembly required)

## Offline Capability
- All models bundled locally
- No Python backend required
- No internet connection needed (except for chatbot feature)

## Original Python Files (Legacy)
The `ml/` directory contains the original Python implementation:
- `api_server.py` - Flask API (no longer needed)
- `hybrid_webcam.py` - Original detection logic (ported to JS)
- `doublehand.py` - YOLO webcam script (ported to JS)

These files are kept for reference but are not required for the JavaScript version.
