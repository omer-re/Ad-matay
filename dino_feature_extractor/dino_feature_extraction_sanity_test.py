import os
import cv2
import torch
import numpy as np
import pickle
from PIL import Image
import torchvision.transforms as transforms
from numpy.linalg import norm
import matplotlib.pyplot as plt

"""
This module will test installations and the model you have generated.
"""

# Load the precomputed example features from the .pkl file
with open('../older_versions/example_features.pkl', 'rb') as f:
    example_features = pickle.load(f)


# Define function for cosine similarity
def cosine_similarity(feature1, feature2):
    return np.dot(feature1, feature2) / (norm(feature1) * norm(feature2))


# Load the DINO ResNet-50 model
model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
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
        features = model(input_tensor)  # Extract features using DINO ResNet-50
        features = features.squeeze(0).cpu().numpy()  # Convert to numpy
    return features


# Capture frame from USB camera, MP4 file, or JPG image
def get_frame(source):
    if source == 'camera':
        cap = cv2.VideoCapture(0)  # USB camera
        ret, frame = cap.read()
        if not ret:
            raise Exception("Unable to capture from camera")
        cap.release()
    elif source.endswith('.mp4'):
        cap = cv2.VideoCapture(source)  # MP4 video file
        ret, frame = cap.read()
        if not ret:
            raise Exception(f"Unable to read from video file {source}")
        cap.release()
    elif source.endswith('.jpg') or source.endswith('.png'):
        frame = cv2.imread(source)  # JPG or PNG image
        if frame is None:
            raise Exception(f"Unable to read image file {source}")
    else:
        raise ValueError("Invalid source. Use 'camera', a video file, or an image file.")

    # Convert the frame (numpy array) to a PIL image
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


# Function to visualize the comparison
def visualize_similarity(frame_image, best_match_image_path, similarity_score):
    return
    best_match_image = Image.open(best_match_image_path).convert('RGB')

    # Show the two images side by side with the similarity score
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Display the current frame
    ax[0].imshow(frame_image)
    ax[0].set_title('Current Frame')
    ax[0].axis('off')

    # Display the best matching example
    ax[1].imshow(best_match_image)
    ax[1].set_title(f'Best Match\nSimilarity: {similarity_score:.2f}')
    ax[1].axis('off')

    # Show the plot
    plt.show()


# Main function to compare the frame with example features and visualize
def compare_frame_to_examples(source):
    # Get frame from the chosen source
    frame = get_frame(source)

    # Extract features from the frame
    roi_features = extract_features(frame)

    # Compare extracted ROI features with precomputed example features
    best_match = None
    best_score = -1

    for filename, example_feature in example_features.items():
        similarity_score = cosine_similarity(roi_features, example_feature)
        if similarity_score > best_score:
            best_match = filename
            best_score = similarity_score

    print(f"Best match: {best_match} with similarity score: {best_score}")

    # Visualize the best match
    if best_match:
        best_match_path = os.path.join("../corners/break/right", best_match)  # Adjust path to match example images folder
        # visualize_similarity(frame, best_match_path, best_score)


# Example usage:

# Example usage:
# To use the USB camera: compare_frame_to_examples('camera')
# To use an MP4 file: compare_frame_to_examples('video.mp4')
# To use a JPG file: compare_frame_to_examples('image.jpg')

# compare_frame_to_examples('camera')  # Replace with 'camera', 'video.mp4', or 'image.jpg'
compare_frame_to_examples(r'samples/break/tv_ad_example.jpg')  # Replace with 'camera', 'video.mp4', or 'image.jpg'
