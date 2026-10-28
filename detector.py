import os
import sys
import cv2
import numpy as np

# ---------------------------------------------------------
# CONSTANTS & CONFIGURATION (Project 4 Parameters)
# ---------------------------------------------------------
PROTOTXT_PATH = os.path.join("models", "MobileNetSSD_deploy.prototxt")
MODEL_PATH = os.path.join("models", "MobileNetSSD_deploy.caffemodel")

# Allow passing custom image: python detector.py path/to/image.jpg
# If no argument is passed, default to images/test.jpg
INPUT_IMAGE_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join("images", "test.jpg")

# Generate dynamic output filename based on input image name
base_name = os.path.basename(INPUT_IMAGE_PATH)
OUTPUT_DIR = "output"
OUTPUT_IMAGE_PATH = os.path.join(OUTPUT_DIR, f"detected_{base_name}")

# Gatekeeper Rule: Milestone 3 - Minimum 80% Confidence Filter
CONFIDENCE_THRESHOLD = 0.80

# MobileNet-SSD class labels (Pascal VOC dataset)
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"
]

# Random distinct color palette for each class
np.random.seed(42)
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


def main():
    print("=" * 60)
    print(" DECODELABS PROJECT 4: MACHINE OPTIC NERVE PIPELINE")
    print("=" * 60)

    # ---------------------------------------------------------
    # MILESTONE 1: LIBRARY & MODEL INTEGRATION
    # ---------------------------------------------------------
    print("\n[Step 1] Loading pre-trained MobileNet-SSD network...")
    if not os.path.exists(PROTOTXT_PATH) or not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model files missing in models/ directory. Run Step 2 first.")
    
    net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)
    print("-> Model architecture and weights loaded successfully.")

    # ---------------------------------------------------------
    # READ INPUT IMAGE
    # ---------------------------------------------------------
    print(f"\n[Step 2] Ingesting visual data from: {INPUT_IMAGE_PATH}")
    image = cv2.imread(INPUT_IMAGE_PATH)
    if image is None:
        raise FileNotFoundError(f"Could not load image at {INPUT_IMAGE_PATH}. Please check the file path.")

    (h, w) = image.shape[:2]
    print(f"-> Input Image Dimensions: {w}x{h} (3 Color Channels: BGR)")

    # ---------------------------------------------------------
    # MILESTONE 2: PRE-PROCESSING INTEGRITY (4D Blob Construction)
    # ---------------------------------------------------------
    print("\n[Step 3] Constructing 4D Input Blob (Pre-processing)...")
    blob = cv2.dnn.blobFromImage(
        image=cv2.resize(image, (300, 300)),
        scalefactor=0.007843,
        size=(300, 300),
        mean=127.5
    )
    print(f"-> 4D Blob shape: {blob.shape} [Batch, Channels, Height, Width]")

    # ---------------------------------------------------------
    # FORWARD PASS (INFERENCE)
    # ---------------------------------------------------------
    print("\n[Step 4] Running forward pass through MobileNet-SSD...")
    net.setInput(blob)
    detections = net.forward()
    print("-> Inference complete. Analyzing probability matrix...")

    # ---------------------------------------------------------
    # MILESTONE 3: ACCURACY BENCHMARKING (The 80% Filter Gatekeeper)
    # ---------------------------------------------------------
    valid_detections_count = 0
    dropped_detections_count = 0

    print(f"\n[Step 5] Applying Gatekeeper Filter (Threshold >= {int(CONFIDENCE_THRESHOLD * 100)}%):")

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence >= CONFIDENCE_THRESHOLD:
            valid_detections_count += 1
            idx = int(detections[0, 0, i, 1])
            class_name = CLASSES[idx]

            # Decode normalized spatial coordinates
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Clamp bounding box coordinates to image boundaries
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w - 1, endX), min(h - 1, endY)

            label = f"{class_name}: {confidence * 100:.2f}%"
            print(f"  [ACCEPTED] {label} at coordinates: ({startX}, {startY}) to ({endX}, {endY})")

            # ---------------------------------------------------------
            # MILESTONE 4: VISUAL CONFIRMATION (Drawing Bounding Box)
            # ---------------------------------------------------------
            color = COLORS[idx]
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
            
            # Position label above the box if room, else below
            y = startY - 10 if startY - 10 > 15 else startY + 20
            cv2.putText(image, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        elif confidence > 0.20:
            dropped_detections_count += 1
            idx = int(detections[0, 0, i, 1])
            class_name = CLASSES[idx]
            print(f"  [DROPPED]  {class_name}: {confidence * 100:.2f}% (Below 80% gatekeeper)")

    print(f"\nSummary: {valid_detections_count} accepted (>= 80%), {dropped_detections_count} dropped (< 80%).")

    # ---------------------------------------------------------
    # SAVE OUTPUT VISUALIZATION
    # ---------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cv2.imwrite(OUTPUT_IMAGE_PATH, image)
    print(f"\n[Step 6] Visual confirmation saved to: {OUTPUT_IMAGE_PATH}")
    print("=" * 60)
    print(" PROJECT 4 PIPELINE EXECUTED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
