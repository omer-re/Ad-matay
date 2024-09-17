import cv2
import numpy as np
from ultralytics import YOLO


def detect_tv_corners(image, mask):
    # Apply morphological operations to refine the mask
    kernel = np.ones((5, 5), np.uint8)
    refined_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours in the refined mask
    contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Find the largest contour (assumed to be the TV)
    largest_contour = max(contours, key=cv2.contourArea)

    # Approximate the contour to a polygon
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    # If we have a quadrilateral, return its corners
    if len(approx) == 4:
        return approx.reshape(-1, 2)

    # If not, we'll use edge detection and line fitting
    roi = cv2.bitwise_and(image, image, mask=refined_mask)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Detect lines using Hough Line Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

    if lines is not None:
        # Extract the endpoints of the lines
        endpoints = []
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            endpoints.append(((x1, y1), (x2, y2)))

        # Find intersections of lines
        corners = []
        for i in range(len(endpoints)):
            for j in range(i + 1, len(endpoints)):
                pt = line_intersection(endpoints[i], endpoints[j])
                if pt is not None:
                    corners.append(pt)

        # If we have at least 4 corners, return the 4 most extreme ones
        if len(corners) >= 4:
            return np.array(find_extreme_corners(corners))

    return None


def line_intersection(line1, line2):
    # Compute the intersection of two lines
    x1, y1 = line1[0]
    x2, y2 = line1[1]
    x3, y3 = line2[0]
    x4, y4 = line2[1]

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None

    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom

    return int(px), int(py)


def find_extreme_corners(points):
    # Find the 4 corners that form the largest quadrilateral
    hull = cv2.convexHull(np.array(points))
    return cv2.approxPolyDP(hull, 0.1 * cv2.arcLength(hull, True), True)


# Load YOLO model
model = YOLO('../yolo_pt_models/yolov8n-seg.pt')

# Initialize video capture
cap = cv2.VideoCapture(0)  # 0 is usually the default USB camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame)

    # Process each detected object
    for r in results:
        boxes = r.boxes
        masks = r.masks

        if masks is not None:
            for seg, box in zip(masks.xy, boxes):
                # Check if the detected object is a TV (class 62 in COCO dataset)
                if box.cls == 62:  # Assuming 62 is the class ID for TV
                    # Create a binary mask
                    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(mask, [seg.astype(int)], 1)

                    # Detect TV corners
                    corners = detect_tv_corners(frame, mask)

                    if corners is not None:
                        # Draw the corners and connect them
                        for i in range(4):
                            cv2.circle(frame, tuple(corners[i]), 3, (0, 255, 0), -1)
                            cv2.line(frame, tuple(corners[i]), tuple(corners[(i + 1) % 4]), (0, 255, 0), 1)

    # Display the result
    cv2.imshow('TV Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()