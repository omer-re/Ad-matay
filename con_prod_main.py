import queue
import time
import cv2
from video_frame_fetcher import VideoFrameFetcher
from tv_detector import TVDetector
from lpr_processor import LPRProcessor
import os

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

def main():
    frame_queue = queue.Queue(maxsize=1)
    roi_queue = queue.Queue(maxsize=1)
    processed_queue = queue.Queue(maxsize=1)  # Queue for processed frames

    # Input source: This can be an IP, a USB camera index, or a file path.
    # input_source = 'http://192.168.1.195:4747/video'  # Example IP camera, change as needed
    input_source = 0  # For USB camera
    # input_source = '/path/to/video.mp4'  # For a video file

    # Determine the source type (IP camera, USB camera, or video file)
    video_source = determine_source(input_source)

    # Initialize workers
    fetcher = VideoFrameFetcher(video_source, frame_queue)
    detector = TVDetector(frame_queue, roi_queue)
    lpr_processor = LPRProcessor(roi_queue, processed_queue)

    # Start workers
    fetcher.start()
    detector.start()
    lpr_processor.start()

    try:
        while True:
            # Non-blocking read for processed frame
            try:
                processed_frame = processed_queue.get_nowait()
                print("Main Thread: Received Processed frame")
            except queue.Empty:
                processed_frame = None

            # Display Processed frame if available
            if processed_frame is not None:
                print("Displaying Processed Frame")
                cv2.imshow('Processed Frame (LPRProcessor Output)', processed_frame)

            # Continuously call cv2.waitKey to ensure OpenCV window updates
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Short sleep to reduce CPU load
            time.sleep(0.01)

    except KeyboardInterrupt:
        fetcher.stop()
        detector.stop()
        lpr_processor.stop()

        fetcher.join()
        detector.join()
        lpr_processor.join()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
