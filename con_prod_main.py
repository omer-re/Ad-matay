import queue
import time
import cv2
import numpy as np
from video_frame_fetcher import VideoFrameFetcher
from tv_detector import TVDetector
from lpr_processor import LPRProcessor
import os
from app_utils import time_measurement

def determine_source(source):
    """
    Determine the type of input source: IP camera, USB camera, or video file.
    """
    if isinstance(source, str):
        if source.startswith('http://') or source.startswith('rtsp://'):
            # IP camera stream
            return source
        elif os.path.isfile(source):
            # Video file
            return source
    elif isinstance(source, int):
        # USB camera (0 for default camera, 1 for secondary, etc.)
        return source
    else:
        raise ValueError(f"Invalid source type: {source}")



def create_black_frame(width, height):
    """
    Creates a blank (black) frame with the given width and height.
    """
    return np.zeros((height, width, 3), dtype=np.uint8)

def add_title(frame, title):
    """
    Adds a title to the top of the frame.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, title, (10, 30), font, 1, (255, 0,0), 2, cv2.LINE_AA)
    return frame

def resize_frame(frame, width, height):
    """
    Resizes the given frame to the specified width and height.
    """
    return cv2.resize(frame, (width, height))

def main():
    frame_queue = queue.Queue(maxsize=1)
    roi_queue = queue.Queue(maxsize=1)
    processed_queue = queue.Queue(maxsize=1)  # Queue for processed frames
    skip_tv_detection = False

    # Input source: This can be an IP, a USB camera index, or a file path.
    input_source = '/home/hailopi/Ad-matay/video_examples/hq_tv_on.mp4'  # Example video file

    # Determine the source type (IP camera, USB camera, or video file)
    video_source = determine_source(input_source)
    if video_source == 'adb':
        skip_tv_detection = True

    # Initialize workers
    fetcher = VideoFrameFetcher(video_source, frame_queue)
    detector = TVDetector(frame_queue, roi_queue)
    lpr_processor = LPRProcessor(roi_queue, processed_queue)

    # Start workers
    fetcher.start()
    detector.start()
    lpr_processor.start()

    target_width, target_height = 640, 480  # Set the standard size for all frames

    try:
        while True:
            # Create black frames in case inputs/outputs are None
            black_frame = create_black_frame(target_width, target_height)

            # Fetcher frames
            fetcher_input = fetcher.input if fetcher.input is not None else black_frame
            fetcher_output = fetcher.output if fetcher.output is not None else black_frame

            # Detector frames
            detector_input = detector.input if detector.input is not None else black_frame
            detector_output = detector.output if detector.output is not None else black_frame

            # LPR Processor frames
            lpr_input = lpr_processor.input if lpr_processor.input is not None else black_frame
            lpr_output = lpr_processor.output if lpr_processor.output is not None else black_frame

            # Resize all frames to the target size (640x480)
            fetcher_input = resize_frame(fetcher_input, target_width, target_height)
            fetcher_output = resize_frame(fetcher_output, target_width, target_height)
            detector_input = resize_frame(detector_input, target_width, target_height)
            detector_output = resize_frame(detector_output, target_width, target_height)
            lpr_input = resize_frame(lpr_input, target_width, target_height)
            lpr_output = resize_frame(lpr_output, target_width, target_height)

            # Add titles to each frame
            fetcher_input = add_title(fetcher_input, 'Fetcher Input')
            fetcher_output = add_title(fetcher_output, 'Fetcher Output')
            detector_input = add_title(detector_input, 'Detector Input')
            detector_output = add_title(detector_output, 'Detector Output')
            lpr_input = add_title(lpr_input, 'LPR Processor Input')
            lpr_output = add_title(lpr_output, 'LPR Processor Output')

            # Combine frames into a grid (3 columns, 2 rows)
            top_row = cv2.hconcat([fetcher_input, detector_input, lpr_input])
            bottom_row = cv2.hconcat([fetcher_output, detector_output, lpr_output])
            combined_frame = cv2.vconcat([top_row, bottom_row])

            # Display the combined frame in a single window
            cv2.imshow('Combined Frames', combined_frame)

            # Continuously call cv2.waitKey to ensure OpenCV window updates
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break  # Exit the loop when 'q' is pressed

            # Short sleep to reduce CPU load
            time.sleep(0.01)

    except KeyboardInterrupt:
        pass  # Handle keyboard interrupt cleanly

    finally:
        # Stop and clean up all worker threads
        fetcher.stop()
        detector.stop()
        lpr_processor.stop()

        fetcher.join()
        detector.join()
        lpr_processor.join()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
