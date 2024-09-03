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
YOLO_ENLRAGED_CORNERS_COLOR = (0, 200, 0)  # Color for scaled corners
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

        # Variables to hold previous segmentation results
        self.previous_segmented_image = None
        self.previous_simplified_corners = None
        self.previous_contours = None
        self.previous_area = 0

    def main_loop(self):
        """Main loop for processing frames from the camera."""
        try:
            while True:
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
                self.largest_tv_mask = largest_tv_mask
                self.largest_tv_area = largest_tv_area
                contours, hierarchy = cv2.findContours(largest_tv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                external_contours = [cnt for cnt, h in zip(contours, hierarchy[0]) if h[3] == -1]
                self.tv_external_countours_yolo = external_contours
                if external_contours:
                    contour = max(external_contours, key=cv2.contourArea)
                    simplified_corners = self.simplify_polygon(contour, max_edges=4)
                    self.tv_last_valid_corners = simplified_corners
                    if len(simplified_corners) == 4:
                        self.tv_last_valid_corners = simplified_corners
                        self.tv_mask_yolo = self.largest_tv_mask

        except Exception as e:
            print(f'Error in find_largest_tv_segment: {e}')

    def set_reference_frame(self):
        """Set the reference frame for feature matching."""
        if self.current_raw_frame is not None:
            self.ref_frame = cv2.bitwise_and(self.current_raw_frame, self.current_raw_frame, mask=self.tv_mask_yolo)
        self.reference_keypoints, self.reference_descriptors = self.feature_detector.detectAndCompute(self.ref_frame, None)

    def has_camera_moved(self):
        """Determine if the camera has moved by comparing the current frame with the reference frame."""
        if self.reference_keypoints is None or self.reference_descriptors is None or self.current_raw_frame is None:
            raise ValueError("Reference frame is not set. Call set_reference_frame() first.")

        new_ref_frame = cv2.bitwise_and(self.current_raw_frame, self.current_raw_frame, mask=self.tv_mask_yolo)
        keypoints, descriptors = self.feature_detector.detectAndCompute(new_ref_frame, None)

        if descriptors is None or len(descriptors) == 0:
            return True

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(self.reference_descriptors, descriptors)

        matches = sorted(matches, key=lambda x: x.distance)
        result = False

        if len(matches) < self.match_threshold:
            result = True
            self.ref_frame = new_ref_frame
            self.reference_descriptors = descriptors
            self.reference_keypoints = keypoints

        return result

    def calculate_frame_diff(self):
        """Calculate the difference between the current frame and the previous frame."""
        if self.previous_raw_frame is None or self.current_raw_frame is None:
            print("16 empty frames")

        frame_diff = cv2.absdiff(self.current_raw_frame, self.previous_raw_frame)
        diff_mask = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
        _, diff_mask = cv2.threshold(diff_mask, 30, 255, cv2.THRESH_BINARY)
        dynamic_overlay = np.zeros_like(self.current_raw_frame)
        dynamic_overlay[diff_mask > 0] = [0, 255, 0]
        self.tv_mask_diff = dynamic_overlay

    def simplify_polygon(self, contour, max_edges=5, epsilon_factor=0.02):
        """Simplify a polygonal contour to reduce the number of points."""
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        simplified_contour = cv2.approxPolyDP(contour, epsilon, True)
        if len(simplified_contour.shape) == 3 and simplified_contour.shape[1] == 1:
            simplified_contour = simplified_contour[:, 0, :]
        return simplified_contour

    def remove_shadows(self):
        """Remove shadows from the current frame by converting to LAB color space and applying CLAHE."""
        lab = cv2.cvtColor(self.current_raw_frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        lab = cv2.merge((l, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def stable_tv_segmentation(self):
        """Stabilize the TV detection by refining the bounding box."""
        self.current_yolo_results = self.segmentation_model(self.current_raw_frame)
        self.find_largest_tv_segment()
        yolo_detection_avg = np.mean(self.tv_last_valid_corners, axis=0)
        self.apply_watershed(self.current_raw_frame, yolo_detection_avg)
        self.get_roi()
        if len(self.tv_last_valid_corners) == 4 and len(self.watershed_corners) == 4:
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
                cv2.drawContours(self.gui_display_frame, [simplified_contour], -1, YOLO_CORNERS_COLOR, 2)

        # Draw the scaled_yolo_corners
        if self.scaled_corners is not None:
            for corner in self.scaled_corners:
                cv2.circle(self.gui_display_frame, tuple(np.int32(corner)), radius=8, color=YOLO_ENLRAGED_CORNERS_COLOR, thickness=-1)

        # Draw the Watershed corners
        for corner in self.watershed_corners:
            cv2.circle(self.gui_display_frame, tuple(np.int32(corner)), radius=5, color=WATERSHED_CORNERS_COLOR, thickness=-1)

    def apply_watershed(self, image, yolo_detection_avg):
        """
        Apply the Watershed algorithm using the average point from YOLO detection as a seed.

        Args:
            image (np.ndarray): The input image on which to apply the Watershed algorithm.
            yolo_detection_avg (np.ndarray): The average point (centroid) of the YOLO-detected TV corners.

        Returns:
            np.ndarray: The segmented image after applying the Watershed algorithm.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply a threshold to create a binary image
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Noise removal (optional but recommended)
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

        # Sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # Finding sure foreground area using distance transform
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

        # Unknown region (where we are not sure if it is foreground or background)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Markers for the Watershed algorithm
        markers = np.zeros_like(gray, dtype=np.int32)

        # Label the sure background
        markers[sure_bg == 255] = 1

        # Label the sure foreground
        markers[sure_fg == 255] = 2

        # Label the unknown region
        markers[unknown == 255] = 0

        # Place the seed point (yolo_detection_avg) into the marker image
        avg_point_int = tuple(np.int32(yolo_detection_avg))
        cv2.circle(markers, avg_point_int, 5, 3, -1)  # Label the seed as '3' in the markers image

        # Apply Watershed
        markers = cv2.watershed(image, markers)

        # Create a binary mask for the specific marker (e.g., marker '3')
        marker_mask = np.zeros_like(gray, dtype=np.uint8)
        marker_mask[markers == 3] = 255

        # Mark the boundaries in red
        segmented_image = image.copy()
        segmented_image[markers == -1] = [0, 0, 255]  # Red color for boundaries

        # Find contours in the marker mask
        contours, _ = cv2.findContours(marker_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        simplified_corners = []

        # Calculate the area of the new segmentation
        new_area = cv2.countNonZero(marker_mask)

        # Check if the new area is larger than 120% of the previous area
        if self.previous_area > 0 and new_area > 1.2 * self.previous_area:
            print("Warning: Segmented area increased by more than 20%. Using previous segmentation.")
            return self.previous_segmented_image, self.previous_simplified_corners, self.previous_contours

        # Update previous area and segmentation results
        self.previous_area = new_area
        self.previous_segmented_image = segmented_image
        self.previous_simplified_corners = simplified_corners
        self.previous_contours = contours

        # Simplify each contour and draw them
        for contour in contours:
            simplified_corners = self.simplify_polygon(contour, max_edges=4)
            for corner in simplified_corners:
                cv2.circle(segmented_image, tuple(np.int32(corner)), radius=5, color=(0, 255, 255), thickness=-1)

        # Draw the average YOLO detection point
        cv2.circle(segmented_image, tuple(np.int32(yolo_detection_avg)), radius=5, color=(255, 0, 255), thickness=3)
        self.watershed_segmented_image, self.watershed_corners, self.watershed_contours = segmented_image, simplified_corners, contours
        if len(simplified_corners) != 4:
            print("Warning: Simplified corners do not form a rectangle.")
            simplified_corners = self.tv_last_valid_corners
            self.watershed_corners = self.tv_last_valid_corners
        return segmented_image, simplified_corners, contours

    def apply_perspective_transform_and_crop(self, target_aspect_ratio=ASPECT_RATIO):
        """Apply perspective transform to the detected TV corners and crop the image."""
        # Scale the watershed corners first
        scaled_corners = self.scale_bounding_polygon(self.watershed_corners, 1.2)
        self.scaled_corners = scaled_corners

        src_pts = np.array(scaled_corners, dtype="float32")

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
        self.apply_perspective_transform_and_crop()

if __name__ == "__main__":
    app = App(cap)
    app.main_loop()
