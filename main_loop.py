import cv2
from ultralytics import YOLO
import numpy as np
from scipy.ndimage import label

# Constants
ALPHA = 0.7
FRAME_INTERVAL = 3
MARGIN_PERCENT = 0.03
KERNEL_SIZE = (11, 11)
SIGMA_X = 0
SIGMA_Y = 0
ASPECT_RATIO = (16, 9)
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
MAX_ACCUMULATED_FRAMES = 5
PINK_COLOR = (255, 105, 180)
YELLOW_COLOR = (255, 255, 0)

SHORT_INTERVAL = 5
MID_INTERVAL = 500
LONG_INTERVAL = 10000

# Open the USB camera (use the correct index for your camera)
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("Error: Could not open video device")
    cap.release()
    exit()

# Set the video frame width and height
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

class CyclicFrameCounter:
    def __init__(self, max_value, interval):
        self.max_value = max_value
        self.interval = interval
        self.current_value = 0

    def increment(self):
        self.current_value = (self.current_value + self.interval) % self.max_value
        return self.current_value

    def reset(self):
        self.current_value = 0

    def get_value(self):
        return self.current_value

class App():
    def __init__(self, cap):
        self.cap = cap
        self.is_tv_stable = False
        self.frame_counter = CyclicFrameCounter(max_value=LONG_INTERVAL, interval=1)
        self.previous_stabilization_frame = None
        self.current_raw_frame = None
        self.previous_raw_frame=self.current_raw_frame
        self.tv_mask_yolo = None
        self.tv_mask_diff = None
        self.tv_last_valid_corners = None
        self.gui_display_frame = None
        self.feature_detector = cv2.ORB_create()
        self.match_threshold = 10
        self.reference_keypoints = None
        self.reference_descriptors = None
        self.largest_tv_mask = None
        self.largest_tv_area = None
        self.segmentation_model = YOLO('yolov8n-seg.pt')
        self.current_yolo_results=None

        # Variables to hold previous segmentation results
        self.previous_segmented_image = None
        self.previous_simplified_corners = None
        self.previous_contours = None
        self.previous_area = 0

    def main_loop(self):
        try:
            while True:
                ret, raw_frame = self.cap.read()
                if not ret:
                    print("Error: Failed to capture image")
                    break

                if self.frame_counter.get_value()<5:
                    self.frame_counter.increment()
                    self.update_current_frame(raw_frame)
                    self.set_reference_frame()

                self.frame_counter.increment()
                self.update_current_frame(raw_frame)
                self.gui_display_frame = self.current_raw_frame.copy()

                # Detect and stabilize TV
                if not self.is_tv_stable:
                    try:
                        self.stable_tv_segmentation()
                    except TypeError as te:
                        print(f'92 still catching detections {te}')

                # Validate TV detection at intervals
                elif self.frame_counter.get_value() % MID_INTERVAL == 0:
                    has_camera_moved = self.has_camera_moved()
                    if has_camera_moved:
                        self.is_tv_stable = False

                # Draw overlays on gui_display_frame
                try:
                    self.draw_overlays()
                    # self.apply_watershed()
                except ValueError as ve:
                    print(f'104 still catching detections {ve}')

                # Display the final frame
                cv2.imshow("TV Detection", self.gui_display_frame)

                # Press 'q' to exit the loop
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("Stopped by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def update_current_frame(self, new_frame):
        self.previous_raw_frame = self.current_raw_frame
        self.current_raw_frame = new_frame

    def get_yolo_detections(self):
        self.current_yolo_results = self.segmentation_model(self.current_raw_frame)
        self.find_largest_tv_segment()

    def find_largest_tv_segment(self):
        largest_tv_area = 0
        largest_tv_mask = None
        try:
            for i, detection in enumerate(self.current_yolo_results[0].boxes):
                if detection.cls == 62:
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
                if external_contours:
                    contour = max(external_contours, key=cv2.contourArea)
                    simplified_corners = self.simplify_polygon(contour, max_edges=4)
                    self.tv_last_valid_corners = simplified_corners

                    if len(simplified_corners) == 4:
                        self.tv_last_valid_corners = simplified_corners
                        self.tv_mask_yolo = self.largest_tv_mask

                    # else - keep using previous corners
        except Exception as e:
            print(f'Error in find_largest_tv_segment: {e}')

    def set_reference_frame(self):
        if self.current_raw_frame is not None:
            self.ref_frame = cv2.bitwise_and(self.current_raw_frame, self.current_raw_frame, mask=self.tv_mask_yolo)
        self.reference_keypoints, self.reference_descriptors = self.feature_detector.detectAndCompute(self.ref_frame, None)

    def has_camera_moved(self):
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
        if self.previous_raw_frame is None or self.current_raw_frame is None:
            print("16 empty frames")

        frame_diff = cv2.absdiff(self.current_raw_frame, self.previous_raw_frame)
        diff_mask = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
        _, diff_mask = cv2.threshold(diff_mask, 30, 255, cv2.THRESH_BINARY)
        dynamic_overlay = np.zeros_like(self.current_raw_frame)
        dynamic_overlay[diff_mask > 0] = [0, 255, 0]
        self.tv_mask_diff = dynamic_overlay

    def simplify_polygon(self, contour, max_edges=5, epsilon_factor=0.02):
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        simplified_contour = cv2.approxPolyDP(contour, epsilon, True)
        if len(simplified_contour.shape) == 3 and simplified_contour.shape[1] == 1:
            simplified_contour = simplified_contour[:, 0, :]
        return simplified_contour

    def remove_shadows(self):
        # Convert to LAB color space
        lab = cv2.cvtColor(self.current_raw_frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L-channel (contrast limited adaptive histogram equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        # Merge the channels and convert back to BGR
        lab = cv2.merge((l, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def stable_tv_segmentation(self):
        """Stabilize the TV detection by refining the bounding box."""
        self.current_yolo_results = self.segmentation_model(self.current_raw_frame)
        self.find_largest_tv_segment()

        if len(self.tv_last_valid_corners) == 4:
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
                cv2.circle(self.gui_display_frame, tuple(np.int32(corner)), radius=5, color=(0, 0, 255), thickness=-1)
        else:
            print("Corners not drawn. Either not found or invalid.")

        # Calculate the average point (centroid)
        yolo_detection_avg = np.mean(self.tv_last_valid_corners, axis=0)

        # Apply Watershed using the centroid as the seed
        segmented_image, simplified_corners, contours = self.apply_watershed(self.current_raw_frame, yolo_detection_avg)

        # Display the segmented image next to the GUI display frame
        cv2.imshow("Watershed Segmentation", segmented_image)

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

        # Color the segmented area in cyan
        segmented_image[markers == 3] = [255, 255, 0]  # Cyan color for the segmented area

        cv2.circle(segmented_image, tuple(np.int32(yolo_detection_avg)), radius=5, color=(255, 0, 255), thickness=3)

        return segmented_image, simplified_corners, contours


if __name__ == "__main__":
    app = App(cap)
    app.main_loop()
