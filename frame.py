import cv2
import os
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

# Load the YOLOv8n-seg model
model = YOLO('yolov8n-seg.pt')

class Frame:
    cap = None
    input_source = None
    cam_source = None
    frame_count = 0
    frame=None
    previous_frame=None
    previous_interval_frame=None

    def __init__(self, input_source=0):
        """
        Initialize the Frame object. If the input_source is a digit, set it as self.cam_source.
        Otherwise, if it is a path, check if the file exists and set it as self.input_source.
        """
        if str(input_source).isdigit():
            self.cam_source = int(input_source)
        elif os.path.isfile(input_source):
            self.input_source = input_source
        else:
            raise ValueError("Invalid input source. Must be a camera index or a valid file path.")

    def capture_frame(self):
        """
        Capture a frame from the video source.
        If cam_source is set, capture from the camera.
        If input_source is a valid file, fetch a frame from the file.
        """
        if self.cap is None:
            if self.cam_source is not None:
                self.cap = cv2.VideoCapture(self.cam_source)
            elif self.input_source is not None:
                self.cap = cv2.VideoCapture(self.input_source)

        if not self.cap.isOpened():
            print("Error: Could not open video device or file.")
            return None, None

        # update constantly
        self.previous_frame= self.frame.copy()

        # update by interval
        if self.frame_count % FRAME_INTERVAL == 0:
            self.previous_interval_frame=self.frame.copy()

        if self.previous_interval_frame is None:
            # prevent NoneType Error
            self.previous_interval_frame=self.previous_frame


        self.ret, self.frame = self.cap.read()
        if self.ret:
            self.frame_count += 1
        return self.ret, self.frame


    def get_current_frame(self):
        return self.frame

    def get_diff_frame(self):
        frame_with_diff_overlay, diff_mask= self.calculate_frame_diff()
        return frame_with_diff_overlay, diff_mask

    def yolo_products(self):
        return self.blurr_and_yolo


    def calculate_frame_diff(self):
        frame_diff = cv2.absdiff(self.frame, self.previous_frame)
        diff_mask = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
        _, diff_mask = cv2.threshold(diff_mask, 30, 255, cv2.THRESH_BINARY)
        dynamic_overlay = np.zeros_like(self.frame)
        dynamic_overlay[diff_mask > 0] = [0, 255, 0]
        self.frame_with_diff_overlay= cv2.addWeighted(self.frame, 1.0, dynamic_overlay, 0.5, 0)
        return self.frame_with_diff_overlay, diff_mask

    def blurr_frame(self, kernel_size=KERNEL_SIZE, sigmaX=SIGMA_X, sigmaY=SIGMA_Y):
        """
        blurring in order to reduce false detections of cables, nails, screws and junk on the wall
        :param kernel_size:
        :param sigmaX:
        :param sigmaY:
        :return:
        """
        self.blurred_frame=cv2.GaussianBlur(self.frame.copy(), kernel_size, sigmaX, sigmaY)
        return self.blurred_frame

    def blurr_and_yolo(self):
        self.blurr_frame()
        self.frame_yolo=model(self.blurr_frame())
        return self.frame_yolo

    def refine_mask(self, mask):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, KERNEL_SIZE)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    def simplify_polygon(self, contour, max_edges=5, epsilon_factor=0.02):
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        simplified_contour = cv2.approxPolyDP(contour, epsilon, True)
        if len(simplified_contour.shape) == 3 and simplified_contour.shape[1] == 1:
            simplified_contour = simplified_contour[:, 0, :]
        return simplified_contour

    def add_margins_to_corners(self, corners, img_shape, margin_percent=MARGIN_PERCENT):
        corners = np.array(corners, dtype="float32")
        min_x = np.min(corners[:, 0])
        max_x = np.max(corners[:, 0])
        min_y = np.min(corners[:, 1])
        max_y = np.max(corners[:, 1])
        width = max_x - min_x
        height = max_y - min_y
        x_margin = min(width * margin_percent, width * 0.03)
        y_margin = min(height * margin_percent, height * 0.03)
        new_corners = []
        for corner in corners:
            x, y = corner
            new_x = np.clip(x + (x - min_x) / width * x_margin - x_margin / 2, 0, img_shape[1] - 1)
            new_y = np.clip(y + (y - min_y) / height * y_margin - y_margin / 2, 0, img_shape[0] - 1)
            new_corners.append([new_x, new_y])
        return np.array(new_corners, dtype="float32")

    def find_largest_tv_segment(self):
        largest_tv_area = 0
        largest_tv_mask = None
        try:
            for i, detection in enumerate(self.frame_yolo[0].boxes):
                if detection.cls == 62:
                    mask = self.frame_yolo[0].masks.data[i].cpu().numpy()
                    mask = (mask * 255).astype('uint8')
                    mask = cv2.resize(mask, (self.frame.shape[1], self.frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                    mask = self.refine_mask(mask)
                    area = np.sum(mask > 0)
                    if area > largest_tv_area:
                        largest_tv_area = area
                        largest_tv_mask = mask

            if largest_tv_mask is not None:
                contours, hierarchy = cv2.findContours(largest_tv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                external_contours = [cnt for cnt, h in zip(contours, hierarchy[0]) if h[3] == -1]
                if external_contours:
                    contour = max(external_contours, key=cv2.contourArea)
                    self.simplified_corners = self.simplify_polygon(contour, max_edges=4)
                    if len(self.simplified_corners) == 4:
                        self.last_valid_corners = self.simplified_corners
                    else:
                        self.simplified_corners = self.last_valid_corners

                    if self.last_valid_corners is not None:
                        self.last_valid_corners = self.add_margins_to_corners(self.last_valid_corners, frame.shape)

        except Exception as e:
            print(f'Error in find_largest_tv_segment: {e}')

        return largest_tv_mask, self.last_valid_corners

    def draw_bounding_polygon(self, color=YELLOW_COLOR, thickness=5):
        try:
            cv2.polylines(self.frame_with_diff_overlay, [np.int32(self.last_valid_corners)], isClosed=True, color=color, thickness=thickness)
        except TypeError as te:
            print(f'Error in draw_bounding_polygon: {te}')

    def apply_overlay(self,overlay, mask, frame):
        """
        utility funtion, therefore I'll leave the arguments to allow reuse
        :param overlay:
        :param mask:
        :param frame:
        :return:
        """
        purple_overlay = np.zeros_like(frame)
        purple_overlay[mask > 0] = [128, 0, 128]
        cv2.addWeighted(purple_overlay, ALPHA, overlay, 1 - ALPHA, 0, overlay)

    def get_tv_from_frame(self):
        try:
            self.largest_tv_mask, self.last_valid_corners = self.find_largest_tv_segment()
            if self.largest_tv_mask is not None:
                self.draw_bounding_polygon()
                self.apply_overlay(self.frame_with_diff_overlay, self.largest_tv_mask, self.frame)
        except Exception as e:
            print(f'Error in process_frame_with_model: {e}')
            # will return last valid corners as fallback
        return self.last_valid_corners

    def order_points(self, pts):
        x_sorted = pts[np.argsort(pts[:, 0]), :]
        left_most = x_sorted[:2, :]
        right_most = x_sorted[2:, :]
        left_most = left_most[np.argsort(left_most[:, 1]), :]
        (tl, bl) = left_most
        right_most = right_most[np.argsort(right_most[:, 1]), :]
        (tr, br) = right_most
        return np.array([tl, tr, br, bl], dtype="float32")

    def apply_perspective_transform_and_crop(self, src_pts, target_aspect_ratio=ASPECT_RATIO):
        src_pts_ordered = self.order_points(src_pts)
        width = self.frame.shape[1]
        height = self.frame.shape[0]

        target_width = width
        target_height = int(width * target_aspect_ratio[1] / target_aspect_ratio[0])

        if target_height > height:
            target_height = height
            target_width = int(height * target_aspect_ratio[0] / target_aspect_ratio[1])

        dst_pts = np.array([[0, 0], [target_width, 0], [target_width, target_height], [0, target_height]],
                           dtype="float32")
        matrix = cv2.getPerspectiveTransform(src_pts_ordered, dst_pts)
        return cv2.warpPerspective(self.frame, matrix, (target_width, target_height))

    def extract_and_transform_bounding_area(self):
        corners=self.last_valid_corners
        if corners is not None:
            src_pts = np.array(corners, dtype="float32")
            return self.apply_perspective_transform_and_crop(self.frame, src_pts)
        return None

    def get_displays(self):
        # Stage 5: Finalize Frame with Overlays and Display
        self.marked_final_frame = cv2.addWeighted(self.frame_with_diff_overlay.copy(), ALPHA, self.frame_with_diff_overlay.copy(), 1 - ALPHA, 0)

        self.marked_final_frame=cv2.polylines( self.marked_final_frame, [np.int32(self.last_valid_corners)], isClosed=True,
                                            color=YELLOW_COLOR, thickness=5)

        self.transformed_area = self.extract_and_transform_bounding_area()


        if self.transformed_area is not None:

            src_pts = np.array(self.last_valid_corners, dtype="float32")
            self.cropped_transformed = self.apply_perspective_transform_and_crop(self.frame, src_pts)


            # Combine images, ensuring all are in BGR format
            combined_img = np.hstack((
                self.final_frame,  # Already in BGR
                self.cropped_transformed,
                self.transformed_area
            ))