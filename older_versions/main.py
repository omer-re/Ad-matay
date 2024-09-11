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

# Load the YOLOv8n-seg model
model = YOLO('../yolov8n-seg.pt')

# Open the USB camera (use the correct index for your camera)
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("Error: Could not open video device")
    cap.release()
    exit()

# Set the video frame width and height
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

def preprocess_frame(frame, kernel_size=KERNEL_SIZE, sigmaX=SIGMA_X, sigmaY=SIGMA_Y):
    return cv2.GaussianBlur(frame, kernel_size, sigmaX, sigmaY)

def simplify_polygon(contour, max_edges=5, epsilon_factor=0.02):
    epsilon = epsilon_factor * cv2.arcLength(contour, True)
    simplified_contour = cv2.approxPolyDP(contour, epsilon, True)
    if len(simplified_contour.shape) == 3 and simplified_contour.shape[1] == 1:
        simplified_contour = simplified_contour[:, 0, :]
    return simplified_contour

def order_points(pts):
    x_sorted = pts[np.argsort(pts[:, 0]), :]
    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]
    left_most = left_most[np.argsort(left_most[:, 1]), :]
    (tl, bl) = left_most
    right_most = right_most[np.argsort(right_most[:, 1]), :]
    (tr, br) = right_most
    return np.array([tl, tr, br, bl], dtype="float32")

def apply_perspective_transform_and_crop(frame, src_pts, target_aspect_ratio=ASPECT_RATIO):
    src_pts_ordered = order_points(src_pts)
    width = frame.shape[1]
    height = frame.shape[0]

    target_width = width
    target_height = int(width * target_aspect_ratio[1] / target_aspect_ratio[0])

    if target_height > height:
        target_height = height
        target_width = int(height * target_aspect_ratio[0] / target_aspect_ratio[1])

    dst_pts = np.array([[0, 0], [target_width, 0], [target_width, target_height], [0, target_height]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(src_pts_ordered, dst_pts)
    return cv2.warpPerspective(frame, matrix, (target_width, target_height))

# def refine_mask(mask):
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, KERNEL_SIZE)
#     return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# def add_margins_to_corners(corners, img_shape, margin_percent=MARGIN_PERCENT):
#     corners = np.array(corners, dtype="float32")
#     min_x = np.min(corners[:, 0])
#     max_x = np.max(corners[:, 0])
#     min_y = np.min(corners[:, 1])
#     max_y = np.max(corners[:, 1])
#     width = max_x - min_x
#     height = max_y - min_y
#     x_margin = min(width * margin_percent, width * 0.03)
#     y_margin = min(height * margin_percent, height * 0.03)
#     new_corners = []
#     for corner in corners:
#         x, y = corner
#         new_x = np.clip(x + (x - min_x) / width * x_margin - x_margin / 2, 0, img_shape[1] - 1)
#         new_y = np.clip(y + (y - min_y) / height * y_margin - y_margin / 2, 0, img_shape[0] - 1)
#         new_corners.append([new_x, new_y])
#     return np.array(new_corners, dtype="float32")

def extract_and_transform_bounding_area(frame, corners):
    if corners is not None:
        src_pts = np.array(corners, dtype="float32")
        return apply_perspective_transform_and_crop(frame, src_pts)
    return None

def calculate_frame_diff(frame, previous_frame):
    frame_diff = cv2.absdiff(frame, previous_frame)
    diff_mask = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
    _, diff_mask = cv2.threshold(diff_mask, 30, 255, cv2.THRESH_BINARY)
    dynamic_overlay = np.zeros_like(frame)
    dynamic_overlay[diff_mask > 0] = [0, 255, 0]
    return cv2.addWeighted(frame, 1.0, dynamic_overlay, 0.5, 0)

def blend_accumulated_frames(accumulated_frames):
    return np.mean(accumulated_frames, axis=0).astype(np.uint8)

def draw_bounding_polygon(overlay, corners, color=YELLOW_COLOR, thickness=5):
    try:
        cv2.polylines(overlay, [np.int32(corners)], isClosed=True, color=color, thickness=thickness)
    except TypeError as te:
        print(f'Error in draw_bounding_polygon: {te}')

def apply_overlay(overlay, mask, frame):
    purple_overlay = np.zeros_like(frame)
    purple_overlay[mask > 0] = [128, 0, 128]
    cv2.addWeighted(purple_overlay, ALPHA, overlay, 1 - ALPHA, 0, overlay)

def process_frame_with_model(model, frame, overlay, last_valid_corners):
    try:
        results = model(frame)
        largest_tv_mask, last_valid_corners = find_largest_tv_segment(results, frame, last_valid_corners)
        if largest_tv_mask is not None:
            draw_bounding_polygon(overlay, last_valid_corners)
            apply_overlay(overlay, largest_tv_mask, frame)
    except Exception as e:
        print(f'Error in process_frame_with_model: {e}')
    return last_valid_corners

# def find_largest_tv_segment(results, frame, last_valid_corners):
#     largest_tv_area = 0
#     largest_tv_mask = None
#     try:
#         for i, detection in enumerate(results[0].boxes):
#             if detection.cls == 62:
#                 mask = results[0].masks.data[i].cpu().numpy()
#                 mask = (mask * 255).astype('uint8')
#                 mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
#                 mask = refine_mask(mask)
#                 area = np.sum(mask > 0)
#                 if area > largest_tv_area:
#                     largest_tv_area = area
#                     largest_tv_mask = mask
#
#         if largest_tv_mask is not None:
#             contours, hierarchy = cv2.findContours(largest_tv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#             external_contours = [cnt for cnt, h in zip(contours, hierarchy[0]) if h[3] == -1]
#             if external_contours:
#                 contour = max(external_contours, key=cv2.contourArea)
#                 simplified_corners = simplify_polygon(contour, max_edges=4)
#                 if len(simplified_corners) == 4:
#                     last_valid_corners = simplified_corners
#                 else:
#                     simplified_corners = last_valid_corners
#
#                 if last_valid_corners is not None:
#                     last_valid_corners = add_margins_to_corners(last_valid_corners, frame.shape)
#
#     except Exception as e:
#         print(f'Error in find_largest_tv_segment: {e}')
#
#     return largest_tv_mask, last_valid_corners


def process_and_display_frames(cap, model):
    last_valid_corners = None
    previous_diff_frame = None
    frame_count = 0

    try:
        while True:
            # Stage 1: Capture Frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image")
                break

            original_frame = frame.copy()  # Keep a copy of the original frame for transformation

            # Stage 2: Update Difference Mask (every FRAME_INTERVAL frames)
            if frame_count % FRAME_INTERVAL == 0:
                if previous_diff_frame is not None:
                    frame_with_overlay = calculate_frame_diff(frame, previous_diff_frame)
                else:
                    frame_with_overlay = frame.copy()
                previous_diff_frame = frame.copy()

            frame_count += 1

            # Stage 3: Preprocess and Apply YOLO Model
            preprocessed_frame = preprocess_frame(frame_with_overlay)
            overlay = frame_with_overlay.copy()

            # Stage 4: Process Frame with YOLO Model and Draw Bounding Polygon
            last_valid_corners = process_frame_with_model(model, preprocessed_frame, overlay, last_valid_corners)

            # Stage 5: Finalize Frame with Overlays and Display
            final_frame = cv2.addWeighted(overlay, ALPHA, frame_with_overlay, 1 - ALPHA, 0)

            if last_valid_corners is not None:
                final_frame = cv2.polylines(final_frame, [np.int32(last_valid_corners)], isClosed=True,
                                            color=YELLOW_COLOR, thickness=5)

                transformed_area = extract_and_transform_bounding_area(original_frame, last_valid_corners)
                if transformed_area is not None:
                    # Convert transformed areas to BGR
                    transformed_area_bgr = cv2.cvtColor(transformed_area, cv2.COLOR_RGB2BGR)

                    src_pts = np.array(last_valid_corners, dtype="float32")
                    cropped_transformed = apply_perspective_transform_and_crop(original_frame, src_pts)
                    cropped_transformed_bgr = cv2.cvtColor(cropped_transformed, cv2.COLOR_RGB2BGR)

                    # Combine images, ensuring all are in BGR format
                    combined_img = np.hstack((
                        final_frame,  # Already in BGR
                        cropped_transformed,
                        transformed_area
                    ))

                    # Display the combined image
                    cv2.imshow('combined_img', combined_img)
                else:
                    # If transformation fails, just show the final frame
                    cv2.imshow('final', final_frame)

            else:
                # Display the final frame without transformations
                cv2.imshow('final', final_frame)

            # Press 'q' to exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()

# Run the main processing function
process_and_display_frames(cap, model)