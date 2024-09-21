"""
Constants and paths are concentrated here to allow easy configuration for the user

"""

MIN_LOOP_DELAY=0.01
JUMP_SIZE=100
APP_TIMING_LOG_FILE='app_timings.txt'

ASPECT_RATIO = (16, 9)  # Example aspect ratio for cropping (you can modify this)
LOOP_DELAY=0.05
YOLO_PT_SEG_MODEL_PATH='yolo_pt_models/yolov8n-seg.pt'

ICON_RIGHT_FOLDER = "/home/hailopi/Ad-matay/corners/break/right"
ICON_LEFT_FOLDER = "/home/hailopi/Ad-matay/corners/break/left"

DINO_FEATURES_LEFT_PATH='dino_feature_extractor/example_features_dino_left.pkl'
DINO_FEATURES_RIGHT_PATH='dino_feature_extractor/example_features_dino_right.pkl'
LPR_EXAMPLE_TESTING_VIDEO_PATH='/home/hailopi/Ad-matay/video_input_examples/from_adb/ad2c.mp4'  # Replace with video file path or use 0 for USB camera
CONSECUTIVE_FRAMES_TO_TOGGLE=3
LPRNET_TH=0.6

MAIN_EXAMPLE_TESTING_VIDEO_PATH='/home/hailopi/Ad-matay/video_input_examples/hq_tv_on_ads_dup.mp4'