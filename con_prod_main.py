import queue
import time
import cv2
from video_frame_fetcher import VideoFrameFetcher
from tv_detector import TVDetector
from lpr_processor import LPRProcessor

def main():
    frame_queue = queue.Queue(maxsize=1)
    roi_queue = queue.Queue(maxsize=1)
    processed_queue = queue.Queue(maxsize=1)  # Queue for processed frames

    # Initialize workers
    ip_path = 'http://192.168.1.195:4747/video'
    fetcher = VideoFrameFetcher(ip_path, frame_queue)
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
