import cv2
from ultralytics import YOLO
import numpy as np

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
        self.current_yolo_results=self.segmentation_model(self.current_raw_frame)
        self.current_yolo_results=self.segmentation_model(self.current_raw_frame)
        self.find_largest_tv_segment()
        # self.calculate_frame_diff()

        if len(self.tv_last_valid_corners) == 4:
            self.is_tv_stable = True

    def draw_overlays(self):
        """Draw the detected masks and polygons on the display frame."""
        # Draw the largest_tv_mask in purple
        if self.largest_tv_mask is not None:
            purple_overlay = np.zeros_like(self.gui_display_frame)
            purple_overlay[self.largest_tv_mask > 0] = [128, 0, 128]
            # cv2.addWeighted(purple_overlay, ALPHA, self.gui_display_frame, 1 - ALPHA, 0, self.gui_display_frame)

        # Draw the tv_mask_diff in green
        if self.tv_mask_diff is not None:
            green_overlay = np.zeros_like(self.gui_display_frame)
            green_overlay[self.tv_mask_diff > 0] = [0, 255, 0]
            # cv2.addWeighted(green_overlay, ALPHA, self.gui_display_frame, 1 - ALPHA, 0, self.gui_display_frame)

        # Draw the tv_last_valid_corners as red points
        if self.tv_last_valid_corners is not None and len(self.tv_last_valid_corners) == 4:
            for corner in self.tv_last_valid_corners:
                cv2.circle(self.gui_display_frame, tuple(np.int32(corner)), radius=5, color=(0, 0, 255), thickness=-1)
        else:
            print("Corners not drawn. Either not found or invalid.")


if __name__ == "__main__":
    app = App(cap)
    app.main_loop()
