import cv2
from ultralytics import YOLO
import numpy as np
from scipy.ndimage import label

# Constants for configuration
ALPHA = 0.2  # Alpha value for overlay transparency
FRAME_INTERVAL = 3  # Interval for frame processing
MARGIN_PERCENT = 0.03  # Margin percentage for certain calculations
KERNEL_SIZE = (11, 11)  # Kernel size for image processing
SIGMA_X = 0  # Sigma X for Gaussian Blur
SIGMA_Y = 0  # Sigma Y for Gaussian Blur
ASPECT_RATIO = (16, 9)  # Target aspect ratio for cropping
CAMERA_INDEX = 0  # Index for the camera
CAMERA_WIDTH = 2560  # Width of the camera feed
CAMERA_HEIGHT = 1440  # Height of the camera feed
MAX_ACCUMULATED_FRAMES = 5  # Max frames to accumulate for processing
PINK_COLOR = (255, 105, 180)  # Pink color for overlays
YELLOW_COLOR = (255, 255, 0)  # Yellow color for overlays
YOLO_CORNERS_COLOR = (0, 0, 255)  # Color for YOLO detected corners
YOLO_ENLRAGED_CORNERS_GREEN_COLOR = (0, 200, 0)  # Color for scaled corners
WATERSHED_CORNERS_COLOR = (0, 255, 255)  # Color for Watershed corners
SHORT_INTERVAL = 5  # Short interval for certain operations
MID_INTERVAL = 50  # Mid interval for certain operations
LONG_INTERVAL = 10000  # Long interval for certain operations

# Open the USB camera
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("Error: Could not open video device")
    cap.release()
    exit()

# Set the video format to MJPG
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

"""
check your camera ersolution options using `v4l2-ctl --list-formats-ext`
"""

# Set the video frame width and height
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

# Check if the resolution was set successfully
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
# print(f"Resolution set to: {width}x{height}")

class CyclicFrameCounter:
    """A simple cyclic counter to handle frame intervals."""
    def __init__(self, max_value, interval):
        self.max_value = max_value
        self.interval = interval
        self.current_value = 0

    def increment(self):
        """Increment the counter and loop back if necessary."""
        self.current_value = (self.current_value + self.interval) % self.max_value
        return self.current_value

    def reset(self):
        """Reset the counter to zero."""
        self.current_value = 0

    def get_value(self):
        """Get the current value of the counter."""
        return self.current_value

class App:
    def __init__(self, cap):
        self.cap = cap
        self.is_tv_stable = False  # Flag to determine if TV detection is stable
        self.frame_counter = CyclicFrameCounter(max_value=LONG_INTERVAL, interval=1)
        self.previous_stabilization_frame = None
        self.current_raw_frame = None
        self.previous_raw_frame = self.current_raw_frame
        self.tv_mask_yolo = None
        self.tv_mask_diff = None
        self.tv_last_valid_corners = None
        self.gui_display_frame = None
        self.cropped_transformed = None
        self.feature_detector = cv2.ORB_create()  # Feature detector for image comparison
        self.match_threshold = 10  # Threshold for feature matching
        self.reference_keypoints = None
        self.reference_descriptors = None
        self.largest_tv_mask = None
        self.largest_tv_area = None
        self.segmentation_model = YOLO('yolov8n-seg.pt')  # YOLO segmentation model
        self.current_yolo_results = None
        self.scaled_corners=None
        # Variables to hold previous segmentation results
        self.previous_segmented_image = None
        self.previous_simplified_corners = None
        self.previous_contours = None
        self.previous_area = 0

    def main_loop(self):
        """Main loop for processing frames from the camera."""
        try:
            while True:
                try:
                    ret, raw_frame = self.cap.read()
                    if not ret:
                        print("Error: Failed to capture image")
                        break

                    if self.frame_counter.get_value() < 5:
                        self.frame_counter.increment()
                        self.update_current_frame(raw_frame)
                        self.set_reference_frame()

                    self.frame_counter.increment()
                    self.update_current_frame(raw_frame)
                    self.gui_display_frame = self.current_raw_frame.copy()

                    self.stable_tv_segmentation()  # Stabilize TV detection

                    # Draw overlays on gui_display_frame
                    try:
                        self.draw_overlays()
                    except ValueError as ve:
                        print(f'Error during overlay drawing: {ve}')

                    # Set window names
                    window_name1 = "TV Detection"
                    window_name2 = "cropped_transformed"

                    # Create resizable windows
                    cv2.namedWindow(window_name1, cv2.WINDOW_NORMAL)
                    cv2.namedWindow(window_name2, cv2.WINDOW_NORMAL)

                    # Resize windows to 700x400
                    cv2.resizeWindow(window_name1, 700, 400)
                    cv2.resizeWindow(window_name2, 700, 400)

                    # Position the windows next to each other
                    cv2.moveWindow(window_name1, 100, 100)  # Move first window to position (100, 100)
                    cv2.moveWindow(window_name2, 810, 100)  # Move second window to the right of the first one

                    # Display the final frame
                    cv2.imshow(window_name1, self.gui_display_frame)
                    cv2.imshow(window_name2, self.cropped_transformed)


                    # Press 'q' to exit the loop
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except Exception as ex:
                    continue


        except KeyboardInterrupt:
            print("Stopped by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def update_current_frame(self, new_frame):
        """Update the current frame and maintain the previous frame."""
        self.previous_raw_frame = self.current_raw_frame
        self.current_raw_frame = new_frame

    def get_yolo_detections(self):
        """Run YOLO detection on the current frame."""
        self.current_yolo_results = self.segmentation_model(self.current_raw_frame)
        self.find_largest_tv_segment()

    def scale_bounding_polygon(self, corners, scale_factor):
        """
        Scale the bounding polygon homogeneously while keeping the angles and centroid.

        Args:
            corners (list or np.ndarray): List of corner points (x, y) defining the bounding polygon.
            scale_factor (float): The factor by which to scale the polygon (e.g., 1.2 for 20% increase).

        Returns:
            np.ndarray: Scaled list of corner points (x, y).
        """
        # Convert corners to a NumPy array if not already
        corners = np.array(corners, dtype=np.float32)

        # Calculate the centroid of the polygon
        centroid = np.mean(corners, axis=0)

        # Scale each point relative to the centroid
        scaled_corners = scale_factor * (corners - centroid) + centroid

        return scaled_corners

    def detect_tv_corners(self, image, mask):
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

        # Apply morphological operations to refine the mask
        kernel = np.ones((5, 5), np.uint8)
        refined_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours in the refined mask
        contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, None, None, 0

        # Find the largest contour (assumed to be the TV)
        largest_contour = max(contours, key=cv2.contourArea)
        largest_contour_area = cv2.contourArea(largest_contour)

        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

        # If we have a quadrilateral, return its corners, contours, mask, and area
        if len(approx) == 4:
            corners = approx.reshape(-1, 2)
            return corners, largest_contour, refined_mask, largest_contour_area

        # If not, use edge detection and line fitting
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
                corners = np.array(find_extreme_corners(corners))
                return corners, largest_contour, refined_mask, largest_contour_area

        return None, largest_contour, refined_mask, largest_contour_area

    def find_largest_tv_segment(self):
        """Identify the largest TV segment detected by YOLO."""
        largest_tv_area = 0
        largest_tv_mask = None
        try:
            for i, detection in enumerate(self.current_yolo_results[0].boxes):
                if detection.cls == 62:  # Check if the detected class is a TV
                    mask = self.current_yolo_results[0].masks.data[i].cpu().numpy()
                    mask = (mask * 255).astype('uint8')
                    mask = cv2.resize(mask, (self.current_raw_frame.shape[1], self.current_raw_frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                    area = np.sum(mask > 0)
                    if area > largest_tv_area:
                        largest_tv_area = area
                        largest_tv_mask = mask

            if largest_tv_mask is not None:

                #second layer of segmentation
                # Detect TV corners
                tv_last_valid_corners, tv_external_countours_yolo, largest_tv_mask, largest_tv_area = self.detect_tv_corners(self.current_raw_frame, largest_tv_mask)

                # first time- use whatever you got.
                if self.tv_last_valid_corners is None:
                    self.tv_last_valid_corners, self.tv_external_countours_yolo, self.largest_tv_mask, self.largest_tv_area=tv_last_valid_corners, tv_external_countours_yolo, largest_tv_mask, largest_tv_area

                # if valid- update
                if len(tv_last_valid_corners)==4:
                    self.previous_segmented_image = self.largest_tv_mask
                    self.previous_simplified_corners = self.tv_last_valid_corners
                    self.previous_contours = self.tv_external_countours_yolo
                    self.previous_area = self.largest_tv_area
                    self.tv_last_valid_corners, self.tv_external_countours_yolo, self.largest_tv_mask, self.largest_tv_area=tv_last_valid_corners, tv_external_countours_yolo, largest_tv_mask, largest_tv_area

                # else - use previous

        except Exception as e:
            print(f'Error in find_largest_tv_segment: {e}')


    def simplify_polygon(self, contour, max_edges=5, epsilon_factor=0.02):
        """Simplify a polygonal contour to reduce the number of points."""
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        simplified_contour = cv2.approxPolyDP(contour, epsilon, True)
        if len(simplified_contour.shape) == 3 and simplified_contour.shape[1] == 1:
            simplified_contour = simplified_contour[:, 0, :]
        return simplified_contour

    def stable_tv_segmentation(self):
        """Stabilize the TV detection by refining the bounding box."""
        self.current_yolo_results = self.segmentation_model(self.current_raw_frame)
        self.find_largest_tv_segment()
        # yolo_detection_avg = np.mean(self.tv_last_valid_corners, axis=0)
        # self.apply_watershed(self.current_raw_frame, yolo_detection_avg)
        self.get_roi()
        if len(self.tv_last_valid_corners) == 4:# and len(self.watershed_corners) == 4:
            self.is_tv_stable = True

    def draw_overlays(self):
        """Draw the detected masks and polygons on the display frame."""
        # Draw the largest_tv_mask in purple
        if self.largest_tv_mask is not None:
            purple_overlay = np.zeros_like(self.gui_display_frame)
            purple_overlay[self.largest_tv_mask > 0] = [128, 0, 128]
            cv2.addWeighted(purple_overlay, ALPHA, self.gui_display_frame, 1 - ALPHA, 0, self.gui_display_frame)

        # Draw the tv_mask_diff in green
        if self.tv_mask_diff is not None:
            green_overlay = np.zeros_like(self.gui_display_frame)
            green_overlay[self.tv_mask_diff > 0] = [0, 255, 0]
            cv2.addWeighted(green_overlay, ALPHA, self.gui_display_frame, 1 - ALPHA, 0, self.gui_display_frame)

        # Draw the tv_last_valid_corners as red points
        if self.tv_last_valid_corners is not None and len(self.tv_last_valid_corners) == 4:
            for corner in self.tv_last_valid_corners:
                cv2.circle(self.gui_display_frame, tuple(np.int32(corner)), radius=5, color=YOLO_CORNERS_COLOR, thickness=-1)

        # Draw external contours in red
        if self.tv_external_countours_yolo is not None:
            for contour in self.tv_external_countours_yolo:
                simplified_contour = self.simplify_polygon(contour)  # Simplify the contour
                cv2.drawContours(self.gui_display_frame, [simplified_contour], -1, (100,100,100), 2)

        # Draw the scaled_yolo_corners
        if self.scaled_corners is not None:
            for corner in self.scaled_corners:
                cv2.circle(self.gui_display_frame, tuple(np.int32(corner)), radius=8, color=YOLO_ENLRAGED_CORNERS_GREEN_COLOR, thickness=-1)


    def apply_perspective_transform_and_crop(self, target_aspect_ratio=ASPECT_RATIO):
        """Apply perspective transform to the detected TV corners and crop the image."""
        # Scale the watershed corners first
        # self.scaled_corners = self.scale_bounding_polygon(self.tv_last_valid_corners, 1.1)
        # TODO: scaling is optional
        self.scaled_corners = self.tv_last_valid_corners

        src_pts = np.array(self.scaled_corners, dtype="float32")

        # Compute the convex hull to keep only the most external points
        if len(src_pts) > 4:
            hull = cv2.convexHull(src_pts)
            src_pts = hull[:, 0, :]  # Remove unnecessary dimensions

        def order_points(pts):
            """Order points for perspective transformation."""
            x_sorted = pts[np.argsort(pts[:, 0]), :]
            left_most = x_sorted[:2, :]
            right_most = x_sorted[-2:, :]  # Get the last two points for right_most

            # Sort the points within left_most and right_most based on their y-coordinates
            left_most = left_most[np.argsort(left_most[:, 1]), :]
            (tl, bl) = left_most
            right_most = right_most[np.argsort(right_most[:, 1]), :]
            (tr, br) = right_most

            return np.array([tl, tr, br, bl], dtype="float32")

        src_pts_ordered = order_points(src_pts)
        width = self.current_raw_frame.shape[1]
        height = self.current_raw_frame.shape[0]

        target_width = width
        target_height = int(width * target_aspect_ratio[1] / target_aspect_ratio[0])

        if target_height > height:
            target_height = height
            target_width = int(height * target_aspect_ratio[0] / target_aspect_ratio[1])

        dst_pts = np.array([[0, 0], [target_width, 0], [target_width, target_height], [0, target_height]], dtype="float32")
        matrix = cv2.getPerspectiveTransform(src_pts_ordered, dst_pts)
        self.cropped_transformed = cv2.warpPerspective(self.current_raw_frame, matrix, (target_width, target_height))

    def get_roi(self):
        """Get the Region of Interest by applying perspective transform and cropping."""
        try:
            self.apply_perspective_transform_and_crop()
        except Exception as ex:
            print(f'459 {ex}')

if __name__ == "__main__":
    app = App(cap)
    try:
        app.main_loop()
    except Exception as ex:
        print(f'466 {ex}')
