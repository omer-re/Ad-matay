import cv2
import threading
import queue
import time



class VideoFrameFetcher(threading.Thread):
    def __init__(self, video_source, output_queue):
        super().__init__()
        self.video_source = video_source
        self.output_queue = output_queue
        self.capture = cv2.VideoCapture(video_source, cv2.CAP_V4L2)
        # self.capture = cv2.VideoCapture(video_source)
        self.running = True
        self.last_frame = None
        self.input=None
        self.output=None
        # Set resolution to the maximum supported by the camera (1080p example)
        max_width = 1280  # Set to 1920 for Full HD (1080p)
        max_height = 720  # Set to 1080 for Full HD (1080p)
        # self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, max_width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, max_height)

        # Verify the resolution that is actually being used
        actual_width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"VideoCapture resolution set to: {int(actual_width)}x{int(actual_height)}")

    def run(self):
        while self.running:
            try:
                ret, frame = self.capture.read()
                if not ret:
                    raise Exception("Failed to fetch frame")
                self.last_frame = frame
                self.input=frame
                self.output=frame
                if not self.output_queue.full():
                    self.output_queue.put(frame)
                else:
                    self.output_queue.get()  # Remove old frame if queue is full
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
