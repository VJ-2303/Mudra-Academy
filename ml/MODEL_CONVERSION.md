# Model Conversion Guide

Convert YOLOv8 models (`.pt`) to ONNX format for use in the Mudra Academy web app.

## Quick Start

### 1. Get the Model from Your Friend

Ask them for:
- **`best.pt`** - The trained YOLOv8 model file
- **Class names** - List of mudra names in order (e.g., "Anjali, Chakra, Kartariswastika...")

### 2. Convert to ONNX

**Option A: Using the script (Recommended)**

```bash
cd ml
./convert-model.sh best.pt
```

**Option B: Manual conversion**

```bash
# Install dependencies
pip install ultralytics onnx onnxruntime numpy

# Run conversion
python convert_yolo_to_onnx.py best.pt models/kkpv_web.onnx
```

### 3. Update Class Names

Edit `js/mudra-inference.js` and find the `DOUBLE_HAND_MUDRA_CLASSES` array (around line 220):

```javascript
const DOUBLE_HAND_MUDRA_CLASSES = [
    'Anjali',          // index 0
    'Chakra',          // index 1
    'Kartariswastika', // index 2
    // ... add all class names in the EXACT order from training
];
```

### 4. Test the Application

```bash
npm start
```

---

## Detailed Conversion Options

### Basic Conversion

```python
from ultralytics import YOLO

model = YOLO('best.pt')
model.export(format='onnx', opset=12, simplify=False, imgsz=640)
```

### Custom Input Size

If the model was trained with a different size:

```bash
python convert_yolo_to_onnx.py best.pt --imgsz 416
```

Common sizes: 320, 416, 512, 640, 768, 1024

### Output Location

```bash
# Default: creates best.onnx next to best.pt
python convert_yolo_to_onnx.py best.pt

# Custom output path
python convert_yolo_to_onnx.py best.pt models/kkpv_web.onnx
```

---

## Model Requirements

### Input Specification
| Parameter | Value |
|-----------|-------|
| Format | ONNX |
| Input Shape | [1, 3, 640, 640] (NCHW) |
| Input Type | float32 |
| Value Range | 0.0 - 1.0 (normalized) |
| Color Order | RGB |

### Output Specification
| Parameter | Value |
|-----------|-------|
| Shape | [1, num_classes+4, 8400] |
| Format | [batch, features, predictions] |
| Features | [x, y, w, h, class1, class2, ...] |

### Export Settings
| Setting | Value | Reason |
|---------|-------|--------|
| opset | 12 | Best ONNX Runtime Web compatibility |
| simplify | False | Preserves model accuracy |
| dynamic | False | Fixed size for web performance |
| half | False | FP32 precision for accuracy |
| imgsz | 640 | Standard YOLOv8 input size |

---

## Troubleshooting

### "Model not loading in app"

1. Check ONNX file exists in `ml/models/kkpv_web.onnx`
2. Verify file size is reasonable (10-30 MB typically)
3. Check browser console for errors

### "Wrong mudra detected"

1. Verify class names are in correct order
2. Check if model was trained on same input size
3. Adjust `YOLO_CONF_THRESHOLD` in `mudra-inference.js`

### "No detection happening"

1. Lower `YOLO_CONF_THRESHOLD` (try 0.3)
2. Check if hands are in frame
3. Verify webcam permissions

### "Conversion fails"

1. Ensure model is YOLOv8 format (not YOLOv5)
2. Check ultralytics version: `pip show ultralytics`
3. Try updating: `pip install --upgrade ultralytics`

---

## Verification

### Check ONNX Model in Python

```python
import onnxruntime as ort
import numpy as np

# Load model
session = ort.InferenceSession('models/kkpv_web.onnx')

# Get info
print("Inputs:", session.get_inputs()[0].name, session.get_inputs()[0].shape)
print("Outputs:", session.get_outputs()[0].name, session.get_outputs()[0].shape)

# Test inference
dummy = np.random.randn(1, 3, 640, 640).astype(np.float32)
result = session.run(None, {'images': dummy})
print("Output shape:", result[0].shape)
```

### Expected Output

```
Inputs: images [1, 3, 640, 640]
Outputs: output0 [1, num_classes+4, 8400]
Output shape: (1, 12, 8400)  # Example: 8 classes + 4 bbox coords
```

---

## File Structure After Conversion

```
ml/
├── models/
│   ├── kkpv_web.onnx          # Double-hand YOLO model (NEW)
│   ├── mudra_rf_model.onnx    # Single-hand RF model
│   └── mudra_classes.json     # Single-hand class names
├── convert_yolo_to_onnx.py    # Conversion script
├── convert-model.sh           # Easy conversion wrapper
├── requirements-convert.txt   # Python dependencies
└── MODEL_CONVERSION.md        # This file
```

---

## Tips for Your Friend (The Trainer)

Share these tips to ensure the best model:

### Training Tips
- Use **640x640** input size (or document if different)
- Include varied lighting conditions
- Balance class distribution
- Train for at least 100 epochs
- Use data augmentation

### Export Tips
```python
# Recommended export command for them
from ultralytics import YOLO

model = YOLO('best.pt')
model.export(
    format='onnx',
    opset=12,
    simplify=False,
    imgsz=640,
    half=False
)

# Save class names
print("Class names (copy these):")
print(list(model.names.values()))
```

### What to Send
1. `best.pt` file (or already exported `best.onnx`)
2. Class names list in training order
3. Recommended confidence threshold (tested during training)
4. Input size used during training

---

## Quick Reference

### Conversion Commands

```bash
# One-liner using script
./convert-model.sh best.pt

# Manual with Python
python convert_yolo_to_onnx.py best.pt models/kkpv_web.onnx

# Direct ultralytics export
python -c "from ultralytics import YOLO; YOLO('best.pt').export(format='onnx', opset=12)"
```

### Update Class Names Location

File: `js/mudra-inference.js`
Line: ~220
Array: `DOUBLE_HAND_MUDRA_CLASSES`

### Test After Conversion

```bash
npm start
# Then navigate to Double Hand Detection page
```
