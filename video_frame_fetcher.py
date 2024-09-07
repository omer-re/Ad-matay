import cv2
import threading
import queue
import time

class VideoFrameFetcher(threading.Thread):
    def __init__(self, video_source, output_queue):
        super().__init__()
        self.video_source = video_source
        self.output_queue = output_queue
        self.capture = cv2.VideoCapture(video_source)
        self.running = True
        self.last_frame = None

    def run(self):
        while self.running:
            try:
                ret, frame = self.capture.read()
                if not ret:
                    raise Exception("Failed to fetch frame")
                self.last_frame = frame

                if not self.output_queue.full():
                    self.output_queue.put(frame)
                else:
                    self.output_queue.get()
                    self.output_queue.put(frame)

            except Exception as e:
                print(f"Error fetching frame: {e}")
                if self.last_frame is not None:
                    self.output_queue.put(self.last_frame)

            time.sleep(0.01)

    def stop(self):
        self.running = False
        self.capture.release()

# Independent testing of VideoFrameFetcher
def main():
    output_queue = queue.Queue(maxsize=1)
    video_source = 0  # Change to a video file path if needed

    fetcher = VideoFrameFetcher(video_source, output_queue)
    fetcher.start()

    try:
        while True:
            if not output_queue.empty():
                frame = output_queue.get()
                cv2.imshow('Fetched Frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        fetcher.stop()
        fetcher.join()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()