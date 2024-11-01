import cv2
import threading
import queue
import time
from app_utils import *
from constants import *


class VideoFrameFetcher(threading.Thread):
    def __init__(self, video_source, output_queue):
        super().__init__()
        self.video_source = video_source
        self.output_queue = output_queue
        # Check if the video source is an MP4 file
        if isinstance(self.video_source, str) and self.video_source.endswith('.mp4'):
            print("Using FFMPEG backend for MP4 file")
            self.capture = cv2.VideoCapture(video_source, cv2.CAP_FFMPEG)  # Use FFMPEG for MP4 files
        else:
            print("Using default backend")
            self.capture = cv2.VideoCapture(video_source, cv2.CAP_V4L2)  # Use V4L2 for other sources


        self.running = True
        self.last_frame = None
        self.input=None
        self.output=None
        self.timing_info={}
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


        """
        ADB attributes, currently disabled.
        If we're on adb mode we need to "pair" and propagate the adb flag in order to skip tv_detector expensive steps.
        """
        # self.is_adb = is_adb
        # self.adb_ip = adb_ip  # if adb_ip else "192.168.1.242"
        # self.adb_port = adb_port  # if adb_port else "5555"
        # if isinstance(self.video_source, str):
        #
        #     if self.video_source.endswith('.mp4'):
        #         print("Using FFMPEG backend for MP4 file")
        #         self.capture = cv2.VideoCapture(video_source, cv2.CAP_FFMPEG)
        #     elif self.video_source == "adb" and adb_ip is not None:
        #         print("Using ADB to fetch frames.")
        #         if self.adb_ip:
        #             self.connect_adb_to_device(self.adb_ip, self.adb_port)
        #     else:
        #         print("Using default backend")
        #         self.capture = cv2.VideoCapture(video_source, cv2.CAP_V4L2)
        # else:
        #     print("Using default USB camera")
        #     self.capture = cv2.VideoCapture(video_source)


    def run(self):
        execution_time = 0
        start_time = 0
        while self.running:
            try:
                ret, frame = self.capture.read()
                if not ret:
                    raise Exception("Failed to fetch frame")
                self.last_frame = frame
                self.input=frame
                self.output = add_timing_to_frame(execution_time, frame.copy())

                if not self.output_queue.full():
                    self.output_queue.put(frame)
                else:
                    self.output_queue.get()  # Remove old frame if queue is full
                    self.output_queue.put(frame)

            except Exception as e:
                print(f"Error fetching frame: {e}")
                if self.last_frame is not None:
                    self.output_queue.put(self.last_frame)

            end_time = time.time()
            execution_time = end_time - start_time  # Measure time
            start_time = time.time()
            time.sleep(LOOP_DELAY)

    def restart_video(self):
        """Restart the video from the beginning."""
        if self.capture:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            print("Video restarted.")

    def write_timing_to_file(self, file_name):
        """Write the timing information to a file with the class name."""
        class_name = self.__class__.__name__
        with open(file_name, 'a') as f:
            for func_name, elapsed_time in self.timing_info.items():
                f.write(f"{class_name}:\t{func_name}:\t{elapsed_time:.3f} seconds\n")

    def jump_forward(self, frames=100):
        """
        Skips forward by a specified number of frames in the video.

        Args:
            frames (int): Number of frames to skip forward.

        Raises:
            ValueError: If the video capture is not initialized.
        """
        if self.capture:
            current_frame = int(self.capture.get(cv2.CAP_PROP_POS_FRAMES))
            total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
            new_frame = min(current_frame + frames, total_frames - 1)  # Ensure we don't exceed total frames
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
            print(f"Jumped forward to frame {new_frame}.")

    def jump_backward(self, frames=100):
        """Jump backward by the specified number of frames."""
        if self.capture:
            current_frame = int(self.capture.get(cv2.CAP_PROP_POS_FRAMES))
            new_frame = max(current_frame - frames, 0)  # Ensure we don't go below 0
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
            print(f"Jumped backward to frame {new_frame}.")

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
            time.sleep(LOOP_DELAY)
    except KeyboardInterrupt:
        pass
    finally:
        fetcher.stop()
        fetcher.join()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
