#!/bin/bash
# ==============================================
# YOLO to ONNX Conversion Script
# ==============================================
# 
# This script sets up the environment and converts
# a YOLOv8 model to ONNX format for the web app.
#
# Usage: ./convert-model.sh <model.pt>
# ==============================================

set -e

MODEL_FILE="$1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "========================================"
echo "🎯 YOLO to ONNX Conversion"
echo "========================================"
echo ""

# Check if model file provided
if [ -z "$MODEL_FILE" ]; then
    echo "❌ Error: No model file specified!"
    echo ""
    echo "Usage: ./convert-model.sh <model.pt>"
    echo ""
    echo "Example:"
    echo "  ./convert-model.sh best.pt"
    echo "  ./convert-model.sh ~/Downloads/new_mudra_model.pt"
    exit 1
fi

# Check if model file exists
if [ ! -f "$MODEL_FILE" ]; then
    echo "❌ Error: Model file not found: $MODEL_FILE"
    exit 1
fi

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is required but not installed!"
    echo "   Install Python from: https://www.python.org/"
    exit 1
fi

echo "✅ Found Python: $(python3 --version)"

# Create virtual environment if it doesn't exist
VENV_DIR="$SCRIPT_DIR/.venv-convert"

if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "📦 Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Install/upgrade requirements
echo ""
echo "📥 Installing dependencies (this may take a minute)..."
pip install --quiet --upgrade pip
pip install --quiet -r "$SCRIPT_DIR/requirements-convert.txt"

echo ""
echo "🔄 Starting conversion..."
echo ""

# Run conversion
python3 "$SCRIPT_DIR/convert_yolo_to_onnx.py" "$MODEL_FILE" "$SCRIPT_DIR/models/kkpv_web.onnx"

# Deactivate venv
deactivate

echo ""
echo "========================================"
echo "✅ DONE!"
echo "========================================"
echo ""
echo "The ONNX model has been saved to: ml/models/kkpv_web.onnx"
echo ""
echo "Next steps:"
echo "1. Update class names in js/mudra-inference.js"
echo "2. Test: npm start"
echo ""
