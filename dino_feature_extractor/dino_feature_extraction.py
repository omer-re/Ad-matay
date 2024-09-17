# How to install pytroch that supports numpy for RPI5:
# https://qengineering.eu/install%20pytorch%20on%20raspberry%20pi%205.html

# tester:

# import torch
# import numpy as np
#
# # Create a tensor and convert it to a numpy array
# tensor = torch.tensor([1.0, 2.0, 3.0])
# numpy_array = tensor.numpy()
# tensor_back = torch.from_numpy(numpy_array)
#
# print("PyTorch version:", torch.__version__)
# print("NumPy version:", np.__version__)
# print("Tensor:", tensor)
# print("Numpy Array:", numpy_array)
# print("Tensor Back:", tensor_back)
# print(torch.__config__.show())

import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import pickle

"""
This module creates the pkl file for the feature matching.
You will need to run it once when modifying the items you'd like to compare to.
"""

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
        # Pass the input through the DINO ResNet-50 model directly
        features = model(input_tensor)  # Extract features using the DINO ResNet-50 model
        features = features.squeeze(0).cpu().numpy()  # Convert to numpy
    return features


# Folder containing example images
example_folder = "corners/break/right"
example_features = {}
import tqdm
# Extract features for each example image
# Iterate through all images in the directory and extract features
for filename in tqdm.tqdm(os.listdir(example_folder)):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(example_folder, filename)
        image = Image.open(image_path).convert('RGB')
        features = extract_features(image)
        example_features[filename] = features

# Save the features to a .pkl file
with open('example_features_dino_right.pkl', 'wb') as f:
    pickle.dump(example_features, f)
