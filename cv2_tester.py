import cv2
import numpy as np
from ultralytics import YOLO
import torch
from torchvision import models, transforms
from torchvision.ops import masks_to_boxes


# Function to run Mask R-CNN for segmentation
def segment_tv(image, tv_box):
    # Load pre-trained Mask R-CNN model
    mask_rcnn = models.detection.maskrcnn_resnet50_fpn(pretrained=True).eval()

    # Convert the image to a tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        predictions = mask_rcnn(image_tensor)

    # Extract the masks and bounding boxes from predictions
    masks = predictions[0]['masks']
    labels = predictions[0]['labels']
    boxes = masks_to_boxes(masks).cpu().numpy()

    # Find the mask and box corresponding to the TV within the bounding box
    tv_mask = None
    for i, box in enumerate(boxes):
        if (box[0] >= tv_box[0] and box[1] >= tv_box[1] and
                box[2] <= tv_box[2] and box[3] <= tv_box[3] and
                labels[i].item() == 63):  # Class 63 in COCO is TV
            tv_mask = masks[i, 0].cpu().numpy()
            break

    return tv_mask


# Function to detect and refine TV corners
def detect_and_refine_tv(image):
    # Step 1: YOLOv8 for initial detection
    model = YOLO('yolov8n-seg.pt')
    results = model(image)
    tv_box = None

    for r in results:
        for box in r.boxes:
            if model.names[int(box.cls)] == 'tv':
                tv_box = box.xyxy[0].cpu().numpy().astype(int)
                break

    if tv_box is None:
        return None, None

    # Step 2: Mask R-CNN for segmentation within the TV bounding box
    tv_mask = segment_tv(image, tv_box)

    if tv_mask is None:
        return tv_box, None

    # Step 3: Refine using contour detection and Hough Line Transform
    roi = tv_mask[tv_box[1]:tv_box[3], tv_box[0]:tv_box[2]]
    contours, _ = cv2.findContours(roi.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

        if len(approx) == 4:
            corners = approx.reshape(-1, 2)
            corners[:, 0] += tv_box[0]
            corners[:, 1] += tv_box[1]
            return tv_box, corners

    return tv_box, None


# Main loop to capture frames from the USB camera
def main():
    # Open a connection to the USB camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Detect and refine TV corners in the current frame
        initial_box, refined_corners = detect_and_refine_tv(frame)

        if initial_box is not None:
            # Draw initial YOLOv8 detection (green rectangle)
            cv2.rectangle(frame, (initial_box[0], initial_box[1]), (initial_box[2], initial_box[3]), (0, 255, 0), 2)

            if refined_corners is not None:
                # Draw refined corners (red dots)
                for corner in refined_corners:
                    cv2.circle(frame, tuple(corner), 5, (0, 0, 255), -1)

                # Optional: Draw lines connecting the refined corners (blue lines)
                for i in range(4):
                    cv2.line(frame, tuple(refined_corners[i]), tuple(refined_corners[(i + 1) % 4]), (255, 0, 0), 2)

                print("TV detected and corners refined successfully.")
            else:
                print("TV detected, but corner refinement failed.")
        else:
            print("TV not detected.")

        # Display the frame with marked corners
        cv2.imshow('TV Detection', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
