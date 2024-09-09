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
import timm
import pickle
import tqdm

# Load DINO model (use ResNet50 variant for feature extraction)
model = timm.create_model('resnet50', pretrained=True)
model.eval()  # Set model to evaluation mode

# Transformation for images (resize, normalize)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Folder containing example images
example_folder = "corners/break/right"
example_features = {}

# Extract features for each example image
with torch.no_grad():
    for filename in tqdm.tqdm(os.listdir(example_folder)):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            print(f'Processing {filename}')
            img_path = os.path.join(example_folder, filename)
            img = Image.open(img_path).convert('RGB')
            input_tensor = preprocess(img).unsqueeze(0)  # Create a batch of 1

            # Extract features from the image
            features = model(input_tensor).squeeze(0).numpy()
            example_features[filename] = features

with open('../example_features.pkl', 'wb') as f:
    pickle.dump(example_features, f)