import cv2
import numpy as np
import threading
import queue
import time
from ultralytics import YOLO
from app_utils import *
from constants import *
#         self.segmentation_model = YOLO('yolov8n-seg.pt')  # YOLO segmentation model

class TVDetector(threading.Thread):
    def __init__(self, input_queue, output_queue, is_adb=False):
        """
        Initializes the TV detector with the YOLO segmentation model.
        :param model: The YOLO segmentation model.
        :param input_queue: Queue from which the TVDetector will receive frames.
        :param output_queue: Queue to which the TVDetector will pass frames with ROI marked.
        """
        super().__init__()
        self.segmentation_model = YOLO(YOLO_PT_SEG_MODEL_PATH)  # YOLO segmentation model
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.running = True
        self.last_roi_frame = None  # Holds the previous ROI frame if no new input
        self.tv_last_valid_corners = None  # Holds the last valid TV corners
        self.current_raw_frame = None  # Holds the current raw frame
        self.cropped_transformed = None  # Holds the transformed and cropped frame
        self.input=None
        self.output=None
        self.timing_info = {}


    @time_logger('timing_info')
    def detect_tv(self, frame):
        """
        Detects the largest TV screen in the given frame using YOLOv8 segmentation.

        Args:
            frame (numpy.ndarray): The input video frame for detection.

        Returns:
            numpy.ndarray: Frame with TV region marked, or original frame if no TV is detected.

        Raises:
            ValueError: If the input frame is not a valid numpy array.
        """
        self.current_raw_frame = frame.copy()  # Store the raw frame
        results = self.segmentation_model(frame)  # Perform inference on the frame

        # Extract bounding boxes and masks from results
        boxes = results[0].boxes  # YOLOv8 bounding boxes
        masks = results[0].masks  # YOLOv8 segmentation masks

        if boxes is None or masks is None:
            print("No boxes or masks found in the detection results.")
            return frame  # No detections found

        # Print detected class IDs to debug the correct class ID
        print("Detected class IDs:", [int(box.cls) for box in boxes])

        # Filter results to find the TV class (replace 'tv_class_id' with the correct class ID or label)
        tv_class_id = 62  # Replace this with the correct class ID for TV
        tv_detections = []

        for i, box in enumerate(boxes):
            if int(box.cls) == tv_class_id:  # Compare the class ID
                print(f"Detected TV class with box: {box.xyxy}")
                tv_detections.append((box, masks.data[i] if masks is not None else None))

        if not tv_detections:
            print("No TV detected based on the provided class ID.")
            return frame  # No TV detected, return the original frame

        # For simplicity, let's handle the single detection case directly
        largest_tv, largest_tv_mask = tv_detections[0]

        # Debugging: Check the shape of bounding box tensor
        print(f"Bounding box shape: {largest_tv.xyxy.shape}")

        # Safely extract bounding box coordinates assuming shape (1, 4)
        bbox_np = largest_tv.xyxy.cpu().numpy().flatten()  # Ensure the tensor is flattened
        if len(bbox_np) == 4:
            x1, y1, x2, y2 = map(int, bbox_np)  # Convert coordinates to integers
        else:
            print(f"Invalid bounding box shape: {bbox_np.shape}")
            return frame

        # Draw bounding box on the frame (marking the detected TV)
        # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Process the mask to find corners (if mask is not None)
        if largest_tv_mask is not None:
            mask = largest_tv_mask.cpu().numpy()  # Use the mask from YOLO
            corners, _, refined_mask, _ = self.detect_tv_corners(frame, mask)

            # If corners were found, draw the quadrilateral and store them
            if corners is not None:
                self.tv_last_valid_corners = corners  # Store the last valid corners

                # Ensure corners are valid tuples and draw lines around the corners
                for i in range(4):
                    pt1 = tuple(map(int, corners[i]))
                    pt2 = tuple(map(int, corners[(i + 1) % 4]))
                    cv2.line(frame, pt1, pt2, (150, 255, 0), 4)
            else:
                print("No valid corners found.")
        else:
            print("No mask found for the largest TV detection.")

        # Update self.output with the frame marked with the TV bounding box
        # self.output = frame

        return frame

    @time_logger('timing_info')
    def detect_tv_corners(self, image, mask):
        """
        Detects the corners of the TV in the given frame using its segmentation mask.
        The method uses a series of image processing steps to identify the TV's bounding polygon
        and refine it to ensure the detected shape aligns with the known properties of a TV,
        such as parallel lines and rectangular form.

        Pipeline:
        1. **Mask Preprocessing:**
           - Converts the mask to an 8-bit format for OpenCV operations.
           - Resizes the mask to match the input image dimensions if necessary.
           - Applies morphological operations to clean and refine the mask, making contours more distinct.

        2. **Contour Detection:**
           - Finds all external contours in the refined mask.
           - Selects the largest contour, assuming it corresponds to the TV screen.

        3. **Polygon Approximation:**
           - Approximates the largest contour to a polygon.
           - If the polygon has 4 vertices, it is considered as the TV's bounding quadrilateral and returned directly.

        4. **Line Detection and Intersection (Fallback):**
           - If the contour approximation does not yield a quadrilateral, the method uses a line detection approach.
           - Detects edges in the region of interest (ROI) using the Canny edge detector.
           - Applies the Hough Line Transform to detect straight lines in the ROI.
           - Finds intersections of the detected lines to generate candidate corner points.

        5. **Corner Refinement:**
           - If sufficient candidate corners are found, the method identifies the four extreme points to form the largest quadrilateral.
           - This refined bounding box is returned as the TV's detected corners.

        Return Values:
        - `corners`: The four corners of the detected TV screen as a numpy array.
        - `largest_contour`: The largest contour detected in the refined mask.
        - `refined_mask`: The processed mask used for contour detection.
        - `largest_contour_area`: The area of the largest detected contour.

        If no valid corners are found, the method returns `None` for the corners, along with the other relevant information.

        :param image: The input frame (numpy.ndarray).
        :param mask: The segmentation mask for the TV (numpy.ndarray).
        :return: A tuple containing:
                 - `corners`: (numpy.ndarray) The four corners of the TV screen or `None` if not found.
                 - `largest_contour`: (numpy.ndarray) The largest contour detected in the mask.
                 - `refined_mask`: (numpy.ndarray) The refined segmentation mask.
                 - `largest_contour_area`: (float) The area of the largest contour detected.
        """
        def preprocess_mask(mask, image_size):
            """Convert mask to 8-bit, resize, and apply morphological operations."""
            mask = cv2.convertScaleAbs(mask)  # Convert mask to 8-bit (uint8)
            if mask.shape[:2] != image_size:
                print(f"Resizing mask from {mask.shape[:2]} to {image_size}")
                mask = cv2.resize(mask, (image_size[1], image_size[0]))  # Resize mask to match image size
            kernel = np.ones((5, 5), np.uint8)
            refined_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            return refined_mask

        def find_largest_contour(mask):
            """Find the largest contour in the refined mask."""
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None, 0
            largest_contour = max(contours, key=cv2.contourArea)
            return largest_contour, cv2.contourArea(largest_contour)

        def approximate_polygon(contour):
            """Approximate the contour to a polygon."""
            epsilon = 0.02 * cv2.arcLength(contour, True)
            return cv2.approxPolyDP(contour, epsilon, True)

        def detect_lines_and_intersections(roi):
            """Detect lines using Hough Transform and find their intersections."""
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

            if lines is None:
                return []

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
            intersections = []
            for i in range(len(endpoints)):
                for j in range(i + 1, len(endpoints)):
                    pt = line_intersection(endpoints[i], endpoints[j])
                    if pt is not None:
                        intersections.append(pt)

            return intersections

        def line_intersection(line1, line2):
            """Compute the intersection of two lines."""
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
            """Find the 4 corners that form the largest quadrilateral."""
            hull = cv2.convexHull(np.array(points))
            return cv2.approxPolyDP(hull, 0.1 * cv2.arcLength(hull, True), True)

        # Step 1: Preprocess the mask
        refined_mask = preprocess_mask(mask, image.shape[:2])

        # Step 2: Find the largest contour
        largest_contour, largest_contour_area = find_largest_contour(refined_mask)
        if largest_contour is None:
            return None, None, None, 0

        # Step 3: Approximate the contour to a polygon
        approx = approximate_polygon(largest_contour)
        if len(approx) == 4:
            corners = approx.reshape(-1, 2)
            return corners, largest_contour, refined_mask, largest_contour_area

        # Step 4: If not a quadrilateral, detect lines and find intersections
        roi = cv2.bitwise_and(image, image, mask=refined_mask)
        intersections = detect_lines_and_intersections(roi)

        # Step 5: If we have at least 4 corners, return the 4 most extreme ones
        if len(intersections) >= 4:
            corners = np.array(find_extreme_corners(intersections))
            return corners, largest_contour, refined_mask, largest_contour_area

        # Step 6: use prior knowledge of the TV frame being rectangular to enforce parallel lines
        #TODO: enforce minimal bounding polygon to have parallel lines

        return None, largest_contour, refined_mask, largest_contour_area

    @time_logger('timing_info')
    def apply_perspective_transform_and_crop(self, target_aspect_ratio=ASPECT_RATIO):
        """
        Apply perspective transform to the detected TV corners and crop the image.
        :param target_aspect_ratio: The aspect ratio for the cropped image.
        :return: Cropped and transformed image based on the TV's perspective.
        """
        if self.tv_last_valid_corners is None:
            return None  # No valid corners, can't apply perspective transform

        # Optionally scale the corners, in this case, we keep them as is
        self.scaled_corners = self.tv_last_valid_corners

        src_pts = np.array(self.scaled_corners, dtype="float32")

        # Order the points for perspective transformation
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

        # Compute the target width and height based on the desired aspect ratio
        target_width = width
        target_height = int(width * target_aspect_ratio[1] / target_aspect_ratio[0])

        if target_height > height:
            target_height = height
            target_width = int(height * target_aspect_ratio[0] / target_aspect_ratio[1])

        dst_pts = np.array([[0, 0], [target_width, 0], [target_width, target_height], [0, target_height]], dtype="float32")

        # Apply the perspective transformation
        matrix = cv2.getPerspectiveTransform(src_pts_ordered, dst_pts)
        self.cropped_transformed = cv2.warpPerspective(self.current_raw_frame, matrix, (target_width, target_height))

        return self.cropped_transformed

    def run(self):
        execution_time=0
        start_time=0

        while self.running:
            try:
                # print("TVDetector: run()")
                if not self.input_queue.empty():
                    frame = self.input_queue.get()
                    self.input=frame
                    # print("TVDetector: Frame received from frame_queue")

                    roi_frame = self.detect_tv(frame)  # Detect the TV and mark it on the frame
                    if roi_frame is not None and isinstance(roi_frame, np.ndarray):
                        # print(f"TVDetector: Processed ROI Frame dimensions: {roi_frame.shape}")
                        pass
                    else:
                        print("TVDetector: Invalid ROI Frame detected")

                    # Optionally apply perspective transformation and cropping
                    cropped_frame = self.apply_perspective_transform_and_crop()
                    # Put both the roi_frame and cropped_frame in roi_queue
                    # print("TVDetector: Putting ROI Frame and Cropped Frame in roi_queue")
                    self.output=add_timing_to_frame(execution_time, roi_frame.copy())

                    if not self.output_queue.full():
                        # print("TVDetector: 267 Queue full")
                        self.output_queue.put((roi_frame, cropped_frame))
                    else:
                        self.output_queue.get()  # Remove old frame if queue is full
                        self.output_queue.put((roi_frame, cropped_frame))
                else:
                    print("TVDetector: 274 queue is empty")
            except Exception as e:
                print(f"Error detecting TV: {e}")

            end_time = time.time()
            execution_time = end_time - start_time  # Measure time
            start_time = time.time()
            time.sleep(LOOP_DELAY)

        print("TVDetector stopped")

    def stop(self):
        """
        Gracefully stops the TV detector by stopping the loop.
        """
        self.running = False

    def write_timing_to_file(self, file_name):
        """Write the timing information to a file with the class name."""
        class_name = self.__class__.__name__
        with open(file_name, 'a') as f:
            for func_name, elapsed_time in self.timing_info.items():
                f.write(f"{class_name}:\t\t{func_name}:\t{elapsed_time:.3f} seconds\n")


# Independent testing of TVDetector
def main():
    # Dummy YOLO model placeholder for testing (replace with actual YOLOv8 model instance)
    input_queue = queue.Queue(maxsize=1)
    output_queue = queue.Queue(maxsize=1)

    # Create the TVDetector with the dummy model
    detector = TVDetector(input_queue, output_queue)
    detector.start()

    try:
        # For testing purposes, use VideoCapture to feed frames into the input queue
        # capture = cv2.VideoCapture('http://192.168.1.195:4747/video')  # Change to a video file path if needed
        capture = cv2.VideoCapture(0)  # Change to a video file path if needed
        while True:
            ret, frame = capture.read()
            if not ret:
                break

            if not input_queue.full():
                input_queue.put(frame)

            if not output_queue.empty():
                roi_frame, cropped_frame = output_queue.get()
                if detector.tv_last_valid_corners is not None:
                    for corner in detector.tv_last_valid_corners:
                        cv2.circle(roi_frame, tuple(np.int32(corner)), radius=5, color=(0,0,255),
                               thickness=-1)
                else: print("tv_last_valid_corners is None")

                cv2.imshow('TV Detection', roi_frame)
                if cropped_frame is not None:
                    cv2.imshow('Cropped TV', cropped_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        capture.release()
        detector.stop()
        detector.join()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()