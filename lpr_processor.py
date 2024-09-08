import threading
import queue
import time
import os
import cv2
import torch
import numpy as np
import pickle
from PIL import Image
import torchvision.transforms as transforms
from numpy.linalg import norm
import timm

# Load the ResNet-50 model (same as used for generating the example features)
model = timm.create_model('resnet50', pretrained=True)
model.eval()  # Set model to evaluation mode

# Define transformation for input images (resize and normalize)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to extract features from an input image
def extract_features(image):
    input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = model(input_tensor).squeeze(0).numpy()
    return features

# Function to list all image files from a directory
def get_image_files_from_directory(directory):
    image_extensions = ['.png', '.jpg', '.jpeg']
    return [os.path.join(directory, f) for f in os.listdir(directory) if any(f.endswith(ext) for ext in image_extensions)]

# Old function for SIFT-based matching (kept but not used)
def find_best_match(corner_image, icon_paths, threshold=0.1):
    best_score = 0.0
    for icon_path in icon_paths:
        icon_img = cv2.imread(icon_path, 0)  # Load icon as grayscale
        gray_corner = cv2.cvtColor(corner_image, cv2.COLOR_BGR2GRAY)  # Convert the corner image to grayscale

        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(gray_corner, None)
        kp2, des2 = sift.detectAndCompute(icon_img, None)

        if des1 is None or des2 is None:
            continue

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2)

        good_matches = []
        all_distances = []
        for match1, match2 in matches:
            if match1.distance < 0.7 * match2.distance:
                good_matches.append(match1)
                all_distances.append(match1.distance)

        if not good_matches or not all_distances:
            continue

        num_good_matches = len(good_matches)
        avg_distance = sum(all_distances) / len(all_distances)
        total_matches = len(matches)
        max_distance = max(all_distances)

        if total_matches == 0 or max_distance == 0:
            continue

        confidence_score = (num_good_matches / total_matches) * (1 - avg_distance / max_distance)
        if confidence_score > best_score:
            best_score = confidence_score
        if best_score > threshold:
            break

    return best_score

# Define function for cosine similarity
def cosine_similarity(feature1, feature2):
    return np.dot(feature1, feature2) / (norm(feature1) * norm(feature2))

class LPRProcessor(threading.Thread):
    def __init__(self, input_queue, output_queue=None):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.running = True
        self.last_processed_frame = None
        self.input = None
        self.output = None

        # Load example features into a class attribute
        with open('example_features.pkl', 'rb') as f:
            self.example_features = pickle.load(f)

        # Restore icon paths initialization
        icon_right_folder = "/home/hailopi/Ad-matay/corners/break/right"
        icon_left_folder = "/home/hailopi/Ad-matay/corners/break/left"
        self.icon_right_paths = get_image_files_from_directory(icon_right_folder)
        self.icon_left_paths = get_image_files_from_directory(icon_left_folder)

    def run(self):
        while self.running:
            try:
                if not self.input_queue.empty():
                    roi_frame, cropped_frame = self.input_queue.get()
                    print("LPRProcessor: Frames received from roi_queue")
                    self.input = cropped_frame
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
                    self.output = self.last_processed_frame
            except Exception as e:
                print(f"Error in LPR processing: {e}")

            time.sleep(0.5)

        print("LPRProcessor stopped")

    def run_lprnet(self, cropped_frame, threshold=0.1):
        if cropped_frame is None or not isinstance(cropped_frame, np.ndarray):
            print("Invalid cropped_frame passed to LPRNet. Skipping.")
            return None

        # Get image dimensions and divide into 4x4 grid
        h, w = cropped_frame.shape[:2]
        grid_h, grid_w = h // 4, w // 4

        # Extract top-right and top-left corners as valid image slices
        top_right_corner = cropped_frame[0:grid_h, 3 * grid_w:w]
        top_left_corner = cropped_frame[0:grid_h, 0:grid_w]

        # Convert the corners to PIL images for feature extraction
        top_right_pil = Image.fromarray(cv2.cvtColor(top_right_corner, cv2.COLOR_BGR2RGB))
        top_left_pil = Image.fromarray(cv2.cvtColor(top_left_corner, cv2.COLOR_BGR2RGB))

        # Extract features from the top-right and top-left corners
        top_right_features = extract_features(top_right_pil)
        top_left_features = extract_features(top_left_pil)

        # Perform feature matching with precomputed example features
        matches_right = self.find_best_dino_match(top_right_features, threshold)
        # matches_left = self.find_best_dino_match(top_left_features, threshold)

        font = cv2.FONT_HERSHEY_SIMPLEX

        # Mark the top-right corner
        if matches_right > threshold:
            cv2.rectangle(cropped_frame, (3 * grid_w, 0), (w, grid_h), (0, 255, 0), 3)
            cv2.putText(cropped_frame, f"AD {matches_right}", (3 * grid_w, grid_h), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
            print(">> RIGHT CORNER ADS")
        else:
            cv2.rectangle(cropped_frame, (3 * grid_w, 0), (w, grid_h), (255, 0, 0), 3)
            cv2.putText(cropped_frame, f"Non Ad {matches_right}", (3 * grid_w, grid_h), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
            print(">> RIGHT CORNER CONTENT")

        # # Mark the top-left corner
        # if matches_left > threshold:
        #     cv2.rectangle(cropped_frame, (0, 0), (grid_w, grid_h), (0, 255, 0), 3)
        #     cv2.putText(cropped_frame, f"AD {matches_left}", (grid_w, grid_h), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        #     print(">> LEFT CORNER ADS")
        # else:
        #     print(">> LEFT CORNER CONTENT")
        #     cv2.rectangle(cropped_frame, (0, 0), (grid_w, grid_h), (255, 0, 0), 3)
        #     cv2.putText(cropped_frame, f"Non Ad {matches_left}", (grid_w, grid_h), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

        return cropped_frame

    def find_best_dino_match(self, corner_features, threshold):
        best_score = 0.0
        # Access the precomputed example features using self.example_features
        for filename, example_feature in self.example_features.items():
            similarity_score = cosine_similarity(corner_features, example_feature)
            if similarity_score > best_score:
                best_score = similarity_score
            if best_score > threshold:
                break
        return best_score

    def stop(self):
        self.running = False
        cv2.destroyAllWindows()


# Independent testing of LPRProcessor
def main():
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

if __name__ == "__main__":
    main()
