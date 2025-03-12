# Face Detection Using OpenCV

## Overview
This project implements real-time face detection using OpenCV and Haarcascade classifiers. It captures video from a webcam, detects faces in each frame, and highlights them with a bounding box.

## Requirements
Ensure you have the following dependencies installed:
- Python 3.x
- OpenCV

To install OpenCV, run:
```bash
pip install opencv-python
```

## How It Works
1. Loads the pre-trained Haarcascade model for face detection.
2. Captures frames from the webcam.
3. Converts each frame to grayscale for better detection performance.
4. Detects faces in the frame using the Haarcascade classifier.
5. Draws rectangles around detected faces.
6. Displays the output in real-time.
7. Press 'q' to exit the application.

## Usage
Run the following command in your terminal:
```bash
python face_detection.py
```

## Code Explanation
```python
import cv2

# Load Haarcascade Face Detection Model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
    
    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Features
- Real-time face detection using a webcam.
- Uses OpenCV's Haarcascade classifier for robust face detection.
- Highlights detected faces with a green bounding box.
- Simple and easy to use.

## Future Improvements
- Enhance accuracy with deep learning models like DNN or MTCNN.
- Add facial recognition for identifying specific individuals.
- Implement face tracking for smoother detection.

## License
This project is open-source and free to use.

## Author
Ashish

