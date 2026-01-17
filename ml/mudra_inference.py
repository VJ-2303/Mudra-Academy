import sys
import json
import base64
import cv2
import numpy as np
import logging
from ultralytics import YOLO

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger('mudra_inference')

def main():
    try:
        # Load model
        model_path = 'ml/models/final.pt'
        logger.info(f"Loading model from {model_path}...")
        model = YOLO(model_path)
        logger.info("Model loaded successfully")

        # Ready signal
        print("READY", flush=True)

        # Enforce class names to match the 16-class final.pt model
        CLASSES = [
            'Anjali', 'MATSYA', 'Naagabandha', 'SVASTIKA', 'berunda', 'chakra', 
            'garuda', 'karkota', 'katariswastika', 'katva', 'pasha', 'pushpantha', 
            'sakata', 'shanka', 'shivalinga', 'utsanga'
        ]

        # Process loop
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                
                line = line.strip()
                if not line:
                    continue

                # Expecting base64 string (data:image/jpeg;base64,...)
                if ',' in line:
                    line = line.split(',')[1]
                
                # Decode image
                img_bytes = base64.b64decode(line)
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if img is None:
                    logger.error("Failed to decode image")
                    print(json.dumps({"error": "Image decode failed"}), flush=True)
                    continue

                # Run inference
                # conf=0.5 matches our previous high threshold
                results = model.predict(source=img, conf=0.5, verbose=False)
                
                result_data = {
                    "detected": False
                }

                if results and len(results) > 0:
                    r = results[0]
                    if len(r.boxes) > 0:
                        # Find best detection
                        best_conf = 0
                        best_cls = -1
                        
                        for box in r.boxes:
                            conf = float(box.conf[0])
                            cls_id = int(box.cls[0])
                            
                            if conf > best_conf:
                                best_conf = conf
                                best_cls = cls_id
                        
                        if best_cls != -1 and best_cls < len(CLASSES):
                            class_name = CLASSES[best_cls]
                            result_data = {
                                "detected": True,
                                "name": class_name,
                                "confidence": best_conf
                            }
                
                # Send result
                print(json.dumps(result_data), flush=True)

            except Exception as e:
                logger.error(f"Error during inference loop: {e}")
                print(json.dumps({"error": str(e)}), flush=True)

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
