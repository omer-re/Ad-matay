import threading
import queue
import time


import os
import cv2
import numpy as np

# Function to list all image files from a directory
def get_image_files_from_directory(directory):
    image_extensions = ['.png', '.jpg', '.jpeg']
    return [os.path.join(directory, f) for f in os.listdir(directory) if any(f.endswith(ext) for ext in image_extensions)]

#
# def dummy_lpr_processing(roi_frame):
#     """
#     Dummy function simulating LPRnet. Replace with actual logic.
#     """
#     return roi_frame  # Replace with actual LPR logic



def find_best_match(corner_image, icon_paths, threshold=0.1):
    """
    Compares the given corner image with a list of icon images and returns the best match score.
    :param corner_image: The corner of the current frame (numpy array).
    :param icon_paths: List of file paths to icon images.
    :param threshold: Threshold score for considering a match.
    :return: Best match score for the corner.
    """
    best_score = 0.0
    for icon_path in icon_paths:
        icon_img = cv2.imread(icon_path, 0)  # Load icon as grayscale
        gray_corner = cv2.cvtColor(corner_image, cv2.COLOR_BGR2GRAY)  # Convert the corner image to grayscale

        # Initiate SIFT detector
        sift = cv2.SIFT_create()

        # Find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(gray_corner, None)
        kp2, des2 = sift.detectAndCompute(icon_img, None)

        if des1 is None or des2 is None:
            continue  # Skip if no descriptors found

        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2)

        # Ratio test
        good_matches = []
        all_distances = []

        for match1, match2 in matches:
            if match1.distance < 0.7 * match2.distance:
                good_matches.append(match1)
                all_distances.append(match1.distance)

        # If there are no good matches or distances, skip
        if not good_matches or not all_distances:
            continue

        num_good_matches = len(good_matches)
        avg_distance = sum(all_distances) / len(all_distances)

        total_matches = len(matches)
        max_distance = max(all_distances)

        # Avoid division by zero
        if total_matches == 0 or max_distance == 0:
            continue

        # Calculate confidence score
        confidence_score = (num_good_matches / total_matches) * (1 - avg_distance / max_distance)

        # Update the best score if higher
        if confidence_score > best_score:
            best_score = confidence_score

        # Early exit if match exceeds the threshold
        if best_score > threshold:
            break

    return best_score

class LPRProcessor(threading.Thread):
    def __init__(self, input_queue, output_queue=None):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.running = True
        self.last_processed_frame = None
        self.input=None
        self.output=None
        # Restore icon paths initialization
        icon_right_folder = "/home/hailopi/Ad-matay/corners/break/right"
        icon_left_folder = "/home/hailopi/Ad-matay/corners/break/left"
        self.icon_right_paths = get_image_files_from_directory(icon_right_folder)
        self.icon_left_paths = get_image_files_from_directory(icon_left_folder)

    def run(self):
        while self.running:
            try:
                if not self.input_queue.empty():
                    # Unpack both roi_frame and cropped_frame
                    roi_frame, cropped_frame = self.input_queue.get()
                    print("LPRProcessor: Frames received from roi_queue")
                    self.input=cropped_frame
                    if cropped_frame is None or not isinstance(cropped_frame, np.ndarray):
                        print(f"LPRProcessor: Invalid cropped_frame type. Expected numpy array, got {type(cropped_frame)}")
                        continue

                    print(f"LPRProcessor: Valid cropped_frame received with dimensions {cropped_frame.shape}")
                    # Process the cropped frame
                    self.last_processed_frame = self.run_lprnet(cropped_frame)

                    # Put the processed frame in the output queue for further handling or display
                    if self.output_queue and self.last_processed_frame is not None:
                        if not self.output_queue.full():
                            print("LPRProcessor: Putting processed frame in processed_queue")
                            self.output_queue.put(self.last_processed_frame)
                    self.output=self.last_processed_frame
            except Exception as e:
                print(f"Error in LPR processing: {e}")

            time.sleep(0.5)

        print("LPRProcessor stopped")

    def run_lprnet(self, cropped_frame, icon_right_paths='', icon_left_paths='', threshold=0.1):
        """
        Takes the cropped frame and compares the top-right and top-left corners using feature matching.
        Marks the corner that has a match with a green rectangle, otherwise marks it red.
        """

        if cropped_frame is None or not isinstance(cropped_frame, np.ndarray):
            print("Invalid cropped_frame passed to LPRNet. Skipping.")
            return None

        # Use the class attributes for icon paths if none are provided
        if icon_right_paths == '' or icon_left_paths == '':
            icon_right_paths = self.icon_right_paths
            icon_left_paths = self.icon_left_paths

        # Get image dimensions and divide into 4x4 grid
        h, w = cropped_frame.shape[:2]

        grid_h, grid_w = h // 4, w // 4

        # Extract top-right and top-left corners as valid image slices
        top_right_corner = cropped_frame[0:grid_h, 3 * grid_w:w]
        top_left_corner = cropped_frame[0:grid_h, 0:grid_w]

        # Perform feature matching for the top-right and top-left corners
        matches_right = find_best_match(top_right_corner, icon_right_paths, threshold)
        matches_left = find_best_match(top_left_corner, icon_left_paths, threshold)

        # Mark the top-right corner
        if matches_right > threshold:
            cv2.rectangle(cropped_frame, (3 * grid_w, 0), (w, grid_h), (0, 255, 0), 3)
            print(">> RIGHT CORNER ADS")
        else:
            cv2.rectangle(cropped_frame, (3 * grid_w, 0), (w, grid_h), (0, 0, 255), 3)
            print(">> RIGHT CORNER CONTENT")


        # Mark the top-left corner
        if matches_left > threshold:
            cv2.rectangle(cropped_frame, (0, 0), (grid_w, grid_h), (0, 255, 0), 3)
            print(">> LEFT CORNER ADS")
        else:
            print(">> LEFT CORNER CONTENT")
            cv2.rectangle(cropped_frame, (0, 0), (grid_w, grid_h), (0, 0, 255), 3)

        # Return the processed cropped frame
        return cropped_frame

    def stop(self):
        self.running = False
        cv2.destroyAllWindows()


# Independent testing of LPRProcessor
# Main function
def main():
    # Define the paths to icon folders
    icon_right_folder = "/home/hailopi/Ad-matay/corners/break/right"
    icon_left_folder = "/home/hailopi/Ad-matay/corners/break/left"

    # Get all icon images from the left and right directories
    icon_right_paths = get_image_files_from_directory(icon_right_folder)
    icon_left_paths = get_image_files_from_directory(icon_left_folder)

    # Initialize your application (or object with run_lprnet)
    app = LPRProcessor()

    # Example of loading a frame (you can use live video frames, or image sequences)
    frame = cv2.imread('/path/to/frame_image.png')

    # Run the LPRNet feature matching on the frame
    app.run_lprnet(frame, icon_right_paths, icon_left_paths)

    # Optionally add loop for processing multiple frames (video stream, etc.)

if __name__ == "__main__":
    main()