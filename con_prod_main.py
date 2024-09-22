import cv2
import queue
import time
from video_frame_fetcher import VideoFrameFetcher
from tv_detector import TVDetector
from lpr_processor import LPRProcessor
from app_utils import *
from constants import *

def main():
    frame_queue = queue.Queue(maxsize=1)
    roi_queue = queue.Queue(maxsize=1)
    processed_queue = queue.Queue(maxsize=1)  # Queue for processed frames

    # Input source: This can be an IP, a USB camera index, or a file path.
    input_source = MAIN_EXAMPLE_TESTING_VIDEO_PATH  # Use the constant from constants.py
    # input_source = 0  # for USB cam
    video_output_path = OUTPUT_VIDEO_PATH  # Path where the video will be saved

    # Initialize the workers
    fetcher = VideoFrameFetcher(input_source, frame_queue)
    detector = TVDetector(frame_queue, roi_queue)
    lpr_processor = LPRProcessor(roi_queue, processed_queue)

    # Start the workers (threads)
    fetcher.start()
    detector.start()
    lpr_processor.start()

    # Screen size from constants or manual settings
    screen_width = 1920  # Set manually or based on your screen resolution
    screen_height = 1080  # Set manually or based on your screen resolution

    # Maintain original aspect ratio
    window_width = screen_width // 2
    window_height = int(window_width * (ASPECT_RATIO[1] / ASPECT_RATIO[0]))

    # Set up the named window and resize it
    cv2.namedWindow('Testing GUI', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Testing GUI', window_width, window_height)

    # Initialize the VideoWriter object to export the frames as an MP4 video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec (mp4v for MP4)
    fps = 20  # Frames per second

    # Set the frame size to match the combined frame dimensions
    frame_size = (window_width, window_height)
    video_writer = cv2.VideoWriter(video_output_path, fourcc, fps, frame_size)

    # Check if VideoWriter is opened correctly
    if not video_writer.isOpened():
        print("Error: VideoWriter could not be opened.")
        return

    try:
        while True:
            # Create a black frame with the target dimensions
            black_frame = create_black_frame(TARGET_WIDTH, TARGET_HEIGHT)

            # Fetcher frames
            fetcher_input = fetcher.input if fetcher.input is not None else black_frame
            fetcher_output = fetcher.output if fetcher.output is not None else black_frame

            # Detector frames
            detector_input = detector.input if detector.input is not None else black_frame
            detector_output = detector.output if detector.output is not None else black_frame

            # LPR Processor frames
            lpr_input = lpr_processor.input if lpr_processor.input is not None else black_frame
            lpr_output = lpr_processor.output if lpr_processor.output is not None else black_frame

            # Resize all frames to match the target size (TARGET_WIDTH x TARGET_HEIGHT)
            fetcher_input = resize_frame(fetcher_input, TARGET_WIDTH, TARGET_HEIGHT)
            fetcher_output = resize_frame(fetcher_output, TARGET_WIDTH, TARGET_HEIGHT)
            detector_input = resize_frame(detector_input, TARGET_WIDTH, TARGET_HEIGHT)
            detector_output = resize_frame(detector_output, TARGET_WIDTH, TARGET_HEIGHT)
            lpr_input = resize_frame(lpr_input, TARGET_WIDTH, TARGET_HEIGHT)
            lpr_output = resize_frame(lpr_output, TARGET_WIDTH, TARGET_HEIGHT)

            # Add titles to each frame for clarity
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

            # Ensure the combined frame matches the frame size
            if combined_frame.shape[1] != frame_size[0] or combined_frame.shape[0] != frame_size[1]:
                combined_frame = cv2.resize(combined_frame, frame_size)  # Resize if necessary

            # Display the combined frame in the window
            cv2.imshow('Testing GUI', combined_frame)

            # Write the combined frame to the video file
            video_writer.write(combined_frame)

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

        # Release the VideoWriter object
        video_writer.release()

        cv2.destroyAllWindows()
        # Write timing info to files
        initialize_file(APP_TIMING_LOG_FILE)
        fetcher.write_timing_to_file(APP_TIMING_LOG_FILE)
        detector.write_timing_to_file(APP_TIMING_LOG_FILE)
        lpr_processor.write_timing_to_file(APP_TIMING_LOG_FILE)

        print_last_log_entry(APP_TIMING_LOG_FILE)

if __name__ == "__main__":
    main()
