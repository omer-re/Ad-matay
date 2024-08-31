import cv2
from ultralytics import YOLO
from IPython.display import display, clear_output
from PIL import Image
import numpy as np

# Load the YOLOv8n-seg model
model = YOLO('yolov8n-seg.pt')

# Open the USB camera (usually device 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video device")
    cap.release()
    exit()

# Set the video frame width and height if needed
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break

        # Perform segmentation on the frame
        results = model(frame)
        
        # Draw the segmentation masks on the frame
        annotated_frame = results[0].plot()

        # Convert the frame to RGB format (from BGR)
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Convert the frame to an Image object for display in Jupyter
        img_pil = Image.fromarray(annotated_frame_rgb)

        # Display the image in the notebook
        clear_output(wait=True)  # Clear previous output
        display(img_pil)  # Display the current frame

except KeyboardInterrupt:
    print("Stopped by user")

finally:
    # When everything is done, release the capture
    cap.release()
