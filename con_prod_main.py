import queue
import time
import cv2
from video_frame_fetcher import VideoFrameFetcher
from tv_detector import TVDetector
from lpr_processor import LPRProcessor

def main():
    frame_queue = queue.Queue(maxsize=1)
    roi_queue = queue.Queue(maxsize=1)

    # Initialize workers
    ip_path = 'http://192.168.1.195:4747/video'
    fetcher = VideoFrameFetcher(ip_path, frame_queue)
    detector = TVDetector(frame_queue, roi_queue)
    lpr_processor = LPRProcessor(roi_queue)

    # Start workers
    fetcher.start()
    detector.start()
    lpr_processor.start()
    time.sleep(1)
    try:
        while True:
            # Show the frame that is entering the detector (from frame_queue)
            try:
                raw_frame = frame_queue.get_nowait()  # Non-blocking read from frame_queue
            except queue.Empty:
                raw_frame = None

            # if raw_frame is not None:
            #     print("Displaying Raw Frame (Entering Detector)")
            #     cv2.imshow('Raw Frame (Entering Detector)', raw_frame)

            # Show the ROI frame that is passed from TVDetector to LPRProcessor (from roi_queue)
            try:
                roi_frame, _ = roi_queue.get_nowait()  # Non-blocking read from roi_queue
            except queue.Empty:
                roi_frame = None

            if roi_frame is not None:
                print(f"Displaying ROI Frame (dimensions: {roi_frame.shape})")
                cv2.imshow('ROI Frame (Main Thread)', roi_frame)
            else:
                print("roi frame is None")
            # Continuously call cv2.waitKey to ensure OpenCV window updates
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Short sleep to reduce CPU load, can adjust this based on frame rate
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