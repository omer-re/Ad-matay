import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

# Initialize YOLO model
model = YOLO('yolov8n-seg.pt')

# Constants
BLUE_COLOR = (255, 0, 0)  # Blue color for bounding boxes
FRAME_STACK_SIZE = 20  # Number of frames to average
CAMERA_INDEX = 0  # Camera index


# Function to process the averaged frame with YOLO and draw all detections
def process_frame_with_yolo(average_frame):
    # Convert the averaged frame to a 3-channel image if it's not already
    if len(average_frame.shape) == 2:  # If the image is grayscale
        average_frame_colored = cv2.cvtColor(average_frame, cv2.COLOR_GRAY2BGR)
    else:
        average_frame_colored = average_frame.copy()

    # Run YOLO model on the averaged frame
    results = model(average_frame_colored)

    # Loop through the detected objects and draw their bounding boxes and labels
    for i, detection in enumerate(results[0].boxes):
        # Draw bounding box
        bbox = detection.xyxy.cpu().numpy().astype(int)[0]
        cv2.rectangle(average_frame_colored, (bbox[0], bbox[1]), (bbox[2], bbox[3]), BLUE_COLOR, 2)

        # Extract class ID as an integer
        class_id = int(detection.cls.item())

        # Draw label using the class ID
        label = results[0].names[class_id]
        cv2.putText(average_frame_colored, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLUE_COLOR, 2)

    return average_frame_colored


# Open the camera or video file
cap = cv2.VideoCapture(CAMERA_INDEX)  # Use 0 for the default camera

if not cap.isOpened():
    print("Error: Could not open video device or file.")
    exit()

# Initialize a deque to hold the frames
frame_stack = deque(maxlen=FRAME_STACK_SIZE)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image.")
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Add the new frame to the stack
    frame_stack.append(gray_frame)

    # Compute the average frame
    average_frame = np.mean(np.stack(frame_stack, axis=0), axis=0).astype(np.uint8)

    # Process the averaged frame with YOLO and draw all detections
    processed_frame = process_frame_with_yolo(average_frame)

    # Show the processed frame
    cv2.imshow('Averaged Frame with YOLO Detection', processed_frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
