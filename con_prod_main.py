
import queue
import time
import cv2
from video_frame_fetcher import VideoFrameFetcher
from tv_detector import TVDetector
from lpr_processor import LPRProcessor
from app_utils import *

from constants import *


import pyautogui  # Import to get the screen size

def main():
    frame_queue = queue.Queue(maxsize=1)
    roi_queue = queue.Queue(maxsize=1)
    processed_queue = queue.Queue(maxsize=1)  # Queue for processed frames

    # Input source: This can be an IP, a USB camera index, or a file path.
    input_source = '/home/hailopi/Ad-matay/video_input_examples/hq_tv_on_ads_dup.mp4'  # Example video file
    # input_source = 0  # for USB cam

    # Initialize the workers
    fetcher = VideoFrameFetcher(input_source, frame_queue)
    detector = TVDetector(frame_queue, roi_queue)
    lpr_processor = LPRProcessor(roi_queue, processed_queue)

    # Start the workers (threads)
    fetcher.start()
    detector.start()
    lpr_processor.start()

    # Default screen size if `pyautogui` is not available
    screen_width = 1920  # Set manually or based on your screen resolution
    screen_height = 1080  # Set manually or based on your screen resolution

    # Calculate window width as half of screen width
    window_width = screen_width // 2
    window_height = int(window_width * 0.75)  # Keeping a 4:3 aspect ratio for window height

    target_width, target_height = 640, 480  # Set the standard size for all frames for the testing GUI window only

    # Set up the named window and resize it
    cv2.namedWindow('Testing GUI', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Testing GUI', window_width, window_height)

    try:
        while True:
            black_frame = create_black_frame(target_width, target_height)

            # Fetcher frames
            fetcher_input = fetcher.input if fetcher.input is not None else black_frame
            fetcher_output = fetcher.output if fetcher.output is not None else black_frame

            # Detector frames
            detector_input = detector.input if detector.input is not None else black_frame
            detector_output = detector.output if detector.output is not None else black_frame

            # LPR Processor frames (initialize lpr_output correctly)
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

            # Display the combined frame
            cv2.imshow('Testing GUI', combined_frame)

            # Press 'q' to quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            # Press 'r' to restart the video
            if key == ord('r'):
                fetcher.restart_video()

            # Press '>' to jump forward by frames
            if key == ord('>'):
                fetcher.jump_forward(JUMP_SIZE)

            # Press '<' to jump backward by frames
            if key == ord('<'):
                fetcher.jump_backward(JUMP_SIZE)

            time.sleep(MIN_LOOP_DELAY)  # Reduce CPU load

    except KeyboardInterrupt:
        pass  # Handle interrupt

    finally:
        # Stop all threads and clean up
        fetcher.stop()
        detector.stop()
        lpr_processor.stop()

        fetcher.join()
        detector.join()
        lpr_processor.join()

        cv2.destroyAllWindows()
        # Write timing info to files
        initialize_file(APP_TIMING_LOG_FILE)
        fetcher.write_timing_to_file(APP_TIMING_LOG_FILE)
        detector.write_timing_to_file(APP_TIMING_LOG_FILE)
        lpr_processor.write_timing_to_file(APP_TIMING_LOG_FILE)

        print_last_log_entry(APP_TIMING_LOG_FILE)

if __name__ == "__main__":
    main()


