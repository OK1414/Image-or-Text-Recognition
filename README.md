# Object Detection Using Python, OpenCV, and MobileNet-SSD

**DecodeLabs Industrial Training Kit — Artificial Intelligence Project 4**

## About the Project

This project detects objects in images using Python, OpenCV, and a pre-trained MobileNet-SSD model.

It identifies objects, checks their confidence scores, draws bounding boxes around detections with at least 80% confidence, and saves the output image.

## Technologies Used

- Python
- OpenCV
- NumPy
- MobileNet-SSD

## How It Works

1. Load the pre-trained model.
2. Read the input image.
3. Prepare the image for the model.
4. Run object detection.
5. Accept detections with at least 80% confidence.
6. Draw bounding boxes and labels and save the result.

## Project Structure

```text
project 4/
├── detector.py
├── README.md
├── .gitignore
├── models/
│   ├── MobileNetSSD_deploy.prototxt
│   └── MobileNetSSD_deploy.caffemodel
├── images/
│   ├── test.jpg
│   └── lena.jpg
└── output/
    └── detected_lena.jpg
```

## Installation

Install the required packages:

```powershell
python -m pip install "opencv-python<5" numpy
```

Make sure both model files are inside the `models` folder.

## How to Run

Run the detector with the default image:

```powershell
python detector.py
```

Run it with a specific image:

```powershell
python detector.py "images/lena.jpg"
```

Open the generated image:

```powershell
Start-Process ".\output\detected_lena.jpg"
```

Use the exact output filename printed by your program if it differs.

## Example Result

For one successful run using `images/lena.jpg`:

- **Detected object:** Person
- **Confidence:** 89.70%
- **Confidence threshold:** 80%

Results depend on the input image.

## Key Features

- Image preprocessing
- Deep learning-based object detection
- Confidence-based filtering
- Bounding boxes and labels
- Saving detection results

## Acknowledgment

Created as part of the DecodeLabs Artificial Intelligence Industrial Training Kit.