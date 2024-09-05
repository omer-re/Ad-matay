from video_frame_fetcher import VideoFrameFetcher
from tv_detector import TVDetector
from lpr_processor import LPRProcessor
import queue
import time

def main():
    frame_queue = queue.Queue(maxsize=1)
    roi_queue = queue.Queue(maxsize=1)

    # Initialize workers
    fetcher = VideoFrameFetcher(0, frame_queue)
    detector = TVDetector(frame_queue, roi_queue)
    lpr_processor = LPRProcessor(roi_queue)

    # Start workers
    fetcher.start()
    detector.start()
    lpr_processor.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        fetcher.stop()
        detector.stop()
        lpr_processor.stop()

        fetcher.join()
        detector.join()
        lpr_processor.join()

if __name__ == "__main__":
    main()
