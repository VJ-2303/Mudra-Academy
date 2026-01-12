#!/usr/bin/env python3
"""
YOLO to ONNX Conversion Script for Mudra Academy
=================================================

This script converts a YOLOv8 model (.pt) to ONNX format (.onnx) 
for use in the web application with ONNX Runtime Web.

Usage:
    python convert_yolo_to_onnx.py <input_model.pt> [output_model.onnx]

Examples:
    python convert_yolo_to_onnx.py best.pt
    python convert_yolo_to_onnx.py best.pt kkpv_web.onnx
    python convert_yolo_to_onnx.py new_model.pt models/double_hand_mudra.onnx

Requirements:
    pip install ultralytics onnx onnxruntime

Author: Mudra Academy Team
"""

import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed."""
    missing = []
    
    try:
        import ultralytics
    except ImportError:
        missing.append('ultralytics')
    
    try:
        import onnx
    except ImportError:
        missing.append('onnx')
    
    try:
        import onnxruntime
    except ImportError:
        missing.append('onnxruntime')
    
    if missing:
        print("❌ Missing required packages!")
        print(f"   Please install: pip install {' '.join(missing)}")
        sys.exit(1)
    
    print("✅ All dependencies installed")

def convert_yolo_to_onnx(input_path: str, output_path: str = None, imgsz: int = 640):
    """
    Convert YOLOv8 model to ONNX format.
    
    Args:
        input_path: Path to the .pt model file
        output_path: Optional custom output path (default: same name with .onnx extension)
        imgsz: Input image size (default: 640)
    
    Returns:
        Path to the exported ONNX model
    """
    from ultralytics import YOLO
    
    # Validate input
    input_path = Path(input_path)
    if not input_path.exists():
        print(f"❌ Model file not found: {input_path}")
        sys.exit(1)
    
    if not input_path.suffix == '.pt':
        print(f"⚠️  Warning: Expected .pt file, got {input_path.suffix}")
    
    print(f"📦 Loading model: {input_path}")
    model = YOLO(str(input_path))
    
    # Get model info
    print(f"   Model type: {model.task}")
    if hasattr(model, 'names'):
        print(f"   Classes ({len(model.names)}): {list(model.names.values())}")
    
    print(f"\n🔄 Converting to ONNX format...")
    print(f"   Input size: {imgsz}x{imgsz}")
    print(f"   Opset: 12")
    print(f"   Precision: FP32")
    
    # Export with optimal settings for web deployment
    export_path = model.export(
        format='onnx',
        opset=12,              # Best compatibility with ONNX Runtime Web
        simplify=False,        # Don't simplify - preserves accuracy
        dynamic=False,         # Fixed input size for web
        imgsz=imgsz,          # Input image size
        half=False,            # FP32 precision (most accurate)
    )
    
    print(f"✅ Export complete: {export_path}")
    
    # Move to custom output path if specified
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        import shutil
        shutil.move(export_path, output_path)
        export_path = output_path
        print(f"📁 Moved to: {export_path}")
    
    # Get file size
    size_mb = os.path.getsize(export_path) / (1024 * 1024)
    print(f"📊 File size: {size_mb:.2f} MB")
    
    return str(export_path)

def verify_onnx_model(onnx_path: str):
    """Verify the exported ONNX model."""
    import onnx
    import onnxruntime as ort
    import numpy as np
    
    print(f"\n🔍 Verifying ONNX model...")
    
    # Load and check model
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    print("✅ ONNX model structure is valid")
    
    # Get input/output info
    session = ort.InferenceSession(onnx_path)
    
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    
    print(f"\n📋 Model Information:")
    print(f"   Input name: {input_info.name}")
    print(f"   Input shape: {input_info.shape}")
    print(f"   Input type: {input_info.type}")
    print(f"   Output name: {output_info.name}")
    print(f"   Output shape: {output_info.shape}")
    
    # Test inference
    print(f"\n🧪 Testing inference...")
    input_shape = input_info.shape
    # Handle dynamic dimensions
    test_shape = [1, 3, 640, 640]  # Default test shape
    if isinstance(input_shape[2], int):
        test_shape[2] = input_shape[2]
    if isinstance(input_shape[3], int):
        test_shape[3] = input_shape[3]
    
    dummy_input = np.random.randn(*test_shape).astype(np.float32)
    
    outputs = session.run(None, {input_info.name: dummy_input})
    print(f"✅ Inference successful!")
    print(f"   Output shape: {outputs[0].shape}")
    
    # Extract class count from output
    output_shape = outputs[0].shape
    if len(output_shape) == 3:
        # YOLOv8 output: [batch, num_classes+4, num_predictions]
        num_classes = output_shape[1] - 4
        print(f"   Detected classes: {num_classes}")
    
    return True

def generate_class_names_json(model_path: str, output_path: str = None):
    """Generate a JSON file with class names from the model."""
    from ultralytics import YOLO
    import json
    
    model = YOLO(model_path)
    
    if hasattr(model, 'names'):
        class_names = list(model.names.values())
        
        if output_path is None:
            output_path = Path(model_path).parent / 'double_hand_classes.json'
        
        with open(output_path, 'w') as f:
            json.dump(class_names, f, indent=2)
        
        print(f"\n📝 Class names saved to: {output_path}")
        print(f"   Classes: {class_names}")
        
        return class_names
    else:
        print("⚠️  Could not extract class names from model")
        return None

def print_usage():
    """Print usage instructions."""
    print("""
YOLO to ONNX Conversion Script
==============================

Usage:
    python convert_yolo_to_onnx.py <input_model.pt> [output_model.onnx] [--imgsz SIZE]

Arguments:
    input_model.pt     Path to YOLOv8 model file (.pt)
    output_model.onnx  Optional: Custom output path (default: same folder as input)
    --imgsz SIZE       Optional: Input image size (default: 640)

Examples:
    python convert_yolo_to_onnx.py best.pt
    python convert_yolo_to_onnx.py best.pt models/kkpv_web.onnx
    python convert_yolo_to_onnx.py best.pt --imgsz 416

After conversion:
    1. Copy the .onnx file to ml/models/
    2. Update class names in js/mudra-inference.js
    3. Test in the application
    """)

def main():
    print("=" * 50)
    print("🎯 YOLO to ONNX Conversion for Mudra Academy")
    print("=" * 50)
    print()
    
    # Parse arguments
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    if input_path in ['-h', '--help']:
        print_usage()
        sys.exit(0)
    
    output_path = None
    imgsz = 640
    
    # Parse optional arguments
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--imgsz' and i + 1 < len(sys.argv):
            imgsz = int(sys.argv[i + 1])
            i += 2
        else:
            output_path = sys.argv[i]
            i += 1
    
    # Check dependencies
    check_dependencies()
    print()
    
    # Convert
    onnx_path = convert_yolo_to_onnx(input_path, output_path, imgsz)
    
    # Verify
    verify_onnx_model(onnx_path)
    
    # Extract class names
    generate_class_names_json(input_path)
    
    print("\n" + "=" * 50)
    print("✅ CONVERSION COMPLETE!")
    print("=" * 50)
    print(f"""
Next Steps:
-----------
1. Copy the ONNX model to the models folder:
   cp {onnx_path} ml/models/kkpv_web.onnx

2. Update class names in js/mudra-inference.js:
   Edit the DOUBLE_HAND_MUDRA_CLASSES array

3. Test the application:
   npm start

4. If accuracy seems off, adjust thresholds in mudra-inference.js:
   - YOLO_CONF_THRESHOLD (default: 0.5)
   - YOLO_IOU_THRESHOLD (default: 0.45)
""")

if __name__ == '__main__':
    main()
