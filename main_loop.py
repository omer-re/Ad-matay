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

SHORT_INTERVAL=5
MID_INTERVAL=500
LONG_INTERVAL=10000

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
        """
        Initialize the cyclic frame counter.

        Args:
            max_value (int): The maximum value before cycling back to zero.
            interval (int): The interval at which the counter increments.
        """
        self.max_value = max_value
        self.interval = interval
        self.current_value = 0

    def increment(self):
        """
        Increment the counter based on the interval and cycle back to zero if max_value is reached.
        """
        self.current_value = (self.current_value + self.interval) % self.max_value

        return self.current_value

    def reset(self):
        """
        Reset the counter to zero.
        """
        self.current_value = 0

    def get_value(self):
        """
        Get the current value of the counter.
        """
        return self.current_value

class App():
    def __init__(self, cap):
        self.cap=cap
        self.is_tv_stable= False
        self.frame_counter= CyclicFrameCounter(max_value=LONG_INTERVAL, interval=1)
        self.previous_stabilization_frame=None
        self.current_raw_frame=None
        self.tv_mask_yolo=None
        self.tv_mask_diff=None
        self.tv_last_valid_corners=None

        self.gui_display_frame=None
        """
        Args:
            feature_detector: Feature detector to use (e.g., ORB). If None, ORB will be used by default.
            match_threshold (int): The minimum number of good matches required to consider that the camera has not moved.
        """
        self.feature_detector = cv2.ORB_create()
        self.match_threshold = 10
        # for checking if scene moved
        self.reference_keypoints = None
        self.reference_descriptors = None

        self.largest_tv_mask=None
        self.largest_tv_area=None

        # Load the YOLOv8n-seg model
        self.segmentation_model=YOLO('yolov8n-seg.pt')


    def main_loop(self):
        try:
            while True:
                ret, raw_frame = self.cap.read()
                if not ret:
                    print("Error: Failed to capture image")
                    break

                # every frame

                self.frame_counter.increment()
                # feed pipeline with new frame
                self.update_current_frame(raw_frame)

                # policy for events, what should be called each interval from seldom to often

                # detect ad stabilize TV
                if self.is_tv_stable==False:
                    # run until stabilizing
                    self.stable_tv_segmentation()

                # validate TV detection
                elif self.frame_counter.get_value()%MID_INTERVAL:
                    has_camera_moved=self.has_camera_moved()
                    if has_camera_moved:
                        # re-stable
                        self.is_tv_stable=False

                # check for commercials
                else:
                    # # check for commercials
                    pass

                # Press 'q' to exit the loop
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("Stopped by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def update_current_frame(self, new_frame):
        self.previous_raw_frame=self.current_raw_frame
        self.current_raw_frame=new_frame

    def get_yolo_detections(self):

        self.curret_yolo_results = self.segmentation_model(self.current_raw_frame)

        self.find_largest_tv_segment()


    def find_largest_tv_segment(self):
        largest_tv_area = 0
        largest_tv_mask = None
        try:
            for i, detection in enumerate(self.curret_yolo_results[0].boxes):
                if detection.cls == 62:
                    mask = self.curret_yolo_results[0].masks.data[i].cpu().numpy()
                    mask = (mask * 255).astype('uint8')
                    mask = cv2.resize(mask, (self.current_raw_frame.shape[1], self.current_raw_frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                    # mask = refine_mask(mask)
                    area = np.sum(mask > 0)
                    if area > largest_tv_area:
                        largest_tv_area = area
                        largest_tv_mask = mask

            if largest_tv_mask is not None:
                self.largest_tv_mask=largest_tv_mask
                self.largest_tv_area=largest_tv_area
                contours, hierarchy = cv2.findContours(largest_tv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                external_contours = [cnt for cnt, h in zip(contours, hierarchy[0]) if h[3] == -1]
                if external_contours:
                    contour = max(external_contours, key=cv2.contourArea)
                    simplified_corners = self.simplify_polygon(contour, max_edges=4)
                    if len(simplified_corners) == 4:
                        self.tv_last_valid_corners = simplified_corners
                        self.tv_mask_yolo=self.largest_tv_mask

                    else:
                        print("ERROR on 175: no last valid points")
                    #     self.simplified_corners = self.tv_last_valid_corners

                    # if self.tv_last_valid_corners is not None:
                        # self.tv_last_valid_corners = add_margins_to_corners(self.tv_last_valid_corners, self.current_raw_frame.shape)

        except Exception as e:
            print(f'Error in find_largest_tv_segment: {e}')

        # return largest_tv_mask, self.tv_last_valid_corners


    def set_reference_frame(self):
        """
        Set the reference frame and extract keypoints and descriptors.
        Required for the first run only.
        """
        if self.current_raw_frame is not None:
            # Assuming self.tv_mask_yolo and self.current_raw_frame are already defined in your class:

            # Apply the mask to the current raw frame
            self.ref_frame = cv2.bitwise_and(self.current_raw_frame, self.current_raw_frame, mask=self.tv_mask_yolo)

            # Now, self.ref_frame contains the current frame with the TV area masked out

        self.reference_keypoints, self.reference_descriptors = self.feature_detector.detectAndCompute(self.ref_frame, None)

    def has_camera_moved(self):
        """Check if the camera has moved by comparing the current frame to the reference frame."""
        if self.reference_keypoints is None or self.reference_descriptors is None or self.current_raw_frame is None:
            raise ValueError("Reference frame is not set. Call set_reference_frame() first.")

        new_ref_frame = cv2.bitwise_and(self.current_raw_frame, self.current_raw_frame, mask=self.tv_mask_yolo)

        keypoints, descriptors = self.feature_detector.detectAndCompute(new_ref_frame, None)

        if descriptors is None or len(descriptors) == 0:
            return True  # Consider as moved if no keypoints are found

        # Use BFMatcher to find the best matches between keypoints
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(self.reference_descriptors, descriptors)

        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)
        result=False
        # Check if the number of good matches is below the threshold
        if len(matches) < self.match_threshold:
            result=True
            # update references
            self.ref_frame=new_ref_frame
            self.reference_descriptors=descriptors
            self.reference_keypoints=keypoints

        return result

    def calculate_frame_diff(self):
        frame_diff = cv2.absdiff(self.current_raw_frame, self.previous_raw_frame)
        diff_mask = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
        _, diff_mask = cv2.threshold(diff_mask, 30, 255, cv2.THRESH_BINARY)
        dynamic_overlay = np.zeros_like(self.current_raw_frame)
        dynamic_overlay[diff_mask > 0] = [0, 255, 0]
        self.tv_mask_diff=dynamic_overlay


    def simplify_polygon(self, contour, max_edges=5, epsilon_factor=0.02):
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        simplified_contour = cv2.approxPolyDP(contour, epsilon, True)
        if len(simplified_contour.shape) == 3 and simplified_contour.shape[1] == 1:
            simplified_contour = simplified_contour[:, 0, :]
        return simplified_contour

    def stable_tv_segmentation(self):
        """
        get bounding corners for the TV that is stable.
        use yolo and diff to segment.
        """
        # get raw frame
        # filter for largest TV to prevent false detections
        self.find_largest_tv_segment()
        # mask tv by diff to prevent objects on tv to create false detections
        # set tv mask and corners
        self.tv_mask_diff = None

        # if corners are valid - toggle self.is_tv_stable
        if len(self.tv_last_valid_corners)==4:
            self.is_tv_stable=True











if __name__ == "__main__":
    app=App(cap)
    app.main_loop()
