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
import matplotlib.pyplot as plt
LOOP_DELAY=0.05
# Load the DINO ResNet-50 model
model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
model.eval()  # Set model to evaluation mode

# Define transformation for input images (resize and normalize)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Function to extract features from an input image using DINO ResNet-50
def extract_features(image):
    input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        # Directly pass the input through the model to extract features
        features = model(input_tensor)  # Extract features using DINO ResNet-50 model
        features = features.squeeze(0).cpu().numpy()  # Convert to numpy array
    return features

# Function to list all image files from a directory
def get_image_files_from_directory(directory):
    image_extensions = ['.png', '.jpg', '.jpeg']
    return [os.path.join(directory, f) for f in os.listdir(directory) if any(f.endswith(ext) for ext in image_extensions)]

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

        # Variables to track the result state and buffer count
        self.buffer_count = 0
        self.current_state = "non-ad"  # Initial state
        self.buffer_limit = 3  # Require 5 consecutive frames to confirm state

        # Load the precomputed example features (new DINO features)
        with open('example_features_dino_right.pkl', 'rb') as f_right:
            self.example_features_right = pickle.load(f_right)

        with open('example_features_dino_left.pkl', 'rb') as f_left:
            self.example_features_left = pickle.load(f_left)

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
                        print(
                            f"LPRProcessor: Invalid cropped_frame type. Expected numpy array, got {type(cropped_frame)}")
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

            time.sleep(LOOP_DELAY)

        print("LPRProcessor stopped")

    def run_lprnet(self, cropped_frame, threshold=0.65):
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
        matches_right = self.find_best_dino_match(top_right_features, threshold, side='right')
        matches_left = self.find_best_dino_match(top_left_features, threshold, side='left')

        font = cv2.FONT_HERSHEY_SIMPLEX

        # Determine whether both corners are above the threshold
        if matches_right > threshold and matches_left > threshold:
            self.buffer_count += 1  # Increment buffer count
            if self.buffer_count >= self.buffer_limit:
                self.current_state = "ad"  # Update state to "ad" after 5 frames
        else:
            self.buffer_count = 0  # Reset buffer count if condition not met
            self.current_state = "non-ad"  # State stays as non-ad

        # Mark the top-right corner
        if matches_right > threshold:
            cv2.rectangle(cropped_frame, (3 * grid_w, 0), (w, grid_h), (0, 255, 0), 3)
            cv2.putText(cropped_frame, f"AD {matches_right:.2f}", (3 * grid_w, grid_h), font, 2, (255, 255, 255), 2,
                        cv2.LINE_AA)
            print(">> RIGHT CORNER ADS")
        else:
            cv2.rectangle(cropped_frame, (3 * grid_w, 0), (w, grid_h), (255, 0, 0), 3)
            cv2.putText(cropped_frame, f"Non Ad {matches_right:.2f}", (3 * grid_w, grid_h), font, 2, (255, 255, 255), 2,
                        cv2.LINE_AA)
            print(">> RIGHT CORNER CONTENT")

        # Mark the top-left corner
        if matches_left > threshold:
            cv2.rectangle(cropped_frame, (0, 0), (grid_w, grid_h), (0, 255, 0), 3)
            cv2.putText(cropped_frame, f"AD {matches_left:.2f}", (grid_w, grid_h), font, 2, (255, 255, 255), 2,
                        cv2.LINE_AA)
            print(">> LEFT CORNER ADS")
        else:
            print(">> LEFT CORNER CONTENT")
            cv2.rectangle(cropped_frame, (0, 0), (grid_w, grid_h), (255, 0, 0), 3)
            cv2.putText(cropped_frame, f"Non Ad {matches_left:.2f}", (grid_w, grid_h), font, 2, (255, 255, 255), 2,
                        cv2.LINE_AA)

        # Add the current state to the bottom of the frame
        state_text = f"State: {self.current_state.upper()}"
        state_color= (0,255,0) if self.current_state.upper()=='AD' else (255,0,255)
        cv2.putText(cropped_frame, state_text, (10, h - 20), font, 3, state_color, 5, cv2.LINE_AA)

        return cropped_frame
    def find_best_dino_match(self, corner_features, threshold, side='right'):
        best_score = 0.0
        best_score_ref = ''

        example_features_side = self.example_features_right if side == 'right' else self.example_features_left

        # Access the precomputed example features using self.example_features_right
        for filename, example_feature in example_features_side.items():
            similarity_score = cosine_similarity(corner_features, example_feature)
            if similarity_score > best_score:
                best_score = similarity_score
                best_score_ref = filename
            if best_score > threshold:
                break
        print(f'{best_score_ref=}')
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
