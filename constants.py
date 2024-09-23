"""
Constants and Configuration File

This file contains all the constant values and file paths used throughout the application.
Centralizing these constants allows for easy configuration and maintenance. Modify the values
as needed to adapt the application to different environments and requirements.
"""

# Timing and loop control constants
MIN_LOOP_DELAY = 0.01  # Minimum delay between loop iterations to avoid high CPU usage
JUMP_SIZE = 100  # Number of frames to jump forward/backward in video playback
APP_TIMING_LOG_FILE = 'app_timings.txt'  # Path to log file for application timing
QUEUES_SIZE=2 # 2 will prevent race for resources and deadlocks as writing is separated from reading

# Image and video processing parameters
ASPECT_RATIO = (16, 9)  # Desired aspect ratio for cropping frames
LOOP_DELAY = 0.05  # Standard delay between loop iterations
TARGET_WIDTH = 640
TARGET_HEIGHT = int(TARGET_WIDTH * ASPECT_RATIO[1] / ASPECT_RATIO[0])

# YOLO model configuration
YOLO_PT_SEG_MODEL_PATH = 'yolo_pt_models/yolov8n-seg.pt'  # Path to YOLOv8 segmentation model

# Icon and image directories
ICON_RIGHT_FOLDER = "/home/hailopi/Ad-matay/corners/break/right"  # Folder for right-side icons
ICON_LEFT_FOLDER = "/home/hailopi/Ad-matay/corners/break/left"  # Folder for left-side icons

# Pre-extracted DINO features paths
DINO_FEATURES_LEFT_PATH = 'dino_feature_extractor/example_features_dino_left.pkl'  # DINO features for left side
DINO_FEATURES_RIGHT_PATH = 'dino_feature_extractor/example_features_dino_right.pkl'  # DINO features for right side

# Video file paths for testing
LPR_EXAMPLE_TESTING_VIDEO_PATH = '/home/hailopi/Ad-matay/video_input_examples/from_adb/ad2c.mp4'  # LPR testing video
MAIN_EXAMPLE_TESTING_VIDEO_PATH = '/home/hailopi/Ad-matay/video_input_examples/hq_tv_on_ads_dup.mp4'  # Main testing video
OUTPUT_VIDEO_PATH='/home/hailopi/Ad-matay/demo_images/app_demo_gui_mute.mp4'

# Application parameters
CONSECUTIVE_FRAMES_TO_TOGGLE = 3  # Frames needed to trigger toggle
LPRNET_TH = 0.6  # Threshold for LPR network detection confidence
