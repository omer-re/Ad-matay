import cv2
import threading
import queue
import time
import subprocess
import numpy as np


import time

class VideoFrameFetcher(threading.Thread):
    def __init__(self, video_source, output_queue, is_adb=False, adb_ip=None, adb_port=None):
        super().__init__()
        self.video_source = video_source
        self.output_queue = output_queue
        self.is_adb = is_adb
        self.adb_ip = adb_ip if adb_port else "192.168.1.242"
        self.adb_port = adb_port if adb_port else "5555"
        self.running = True
        self.last_frame = None
        self.input = None
        self.output = None
        self.last_frame_time = time.time()  # Initialize the time tracking for last frame update

        if isinstance(self.video_source, str):
            if self.video_source == "adb":
                print("Using ADB to fetch frames.")
                if self.adb_ip:
                    self.connect_adb_to_device(self.adb_ip, self.adb_port)
            elif self.video_source.endswith('.mp4'):
                print("Using FFMPEG backend for MP4 file")
                self.capture = cv2.VideoCapture(video_source, cv2.CAP_FFMPEG)
            else:
                print("Using default backend")
                self.capture = cv2.VideoCapture(video_source, cv2.CAP_V4L2)
        else:
            print("Using default USB camera")
            self.capture = cv2.VideoCapture(video_source)

    def connect_adb_to_device(self, ip, port):
        """
        Connect to an Android device over TCP/IP.
        :param ip: IP address of the Android device
        :param port: Port number for ADB connection (usually 5555)
        """
        try:
            adb_connect_command = ['adb', 'connect', f'{ip}:{port}']
            subprocess.run(adb_connect_command, check=True)
            print(f"Connected to device at {ip}:{port} over ADB.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to connect to device at {ip}:{port} over ADB: {e}")
            raise

    def run(self):
        if self.video_source == "adb":
            # Handle ADB frame fetching
            while self.running:
                try:
                    frame = self.fetch_adb_frame()
                    if frame is None:
                        raise Exception("Failed to fetch frame from ADB")
                    self.last_frame = frame
                    self.input = frame
                    self.output = frame
                    if not self.output_queue.full():
                        self.output_queue.put(frame)
                    else:
                        self.output_queue.get()  # Remove old frame if queue is full
                        self.output_queue.put(frame)
                except Exception as e:
                    print(f"Error fetching frame from ADB: {e}")
                    if self.last_frame is not None:
                        self.output_queue.put(self.last_frame)
                time.sleep(0.01)
        else:
            # Handle regular frame fetching (camera, mp4, etc.)
            while self.running:
                try:
                    ret, frame = self.capture.read()
                    if not ret:
                        raise Exception("Failed to fetch frame")
                    self.last_frame = frame
                    self.input = frame
                    self.output = frame
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

    def fetch_adb_frame(self):
        """
        Fetch a single frame from an Android device using adb exec-out screencap command.
        """
        try:
            # Run the adb command to fetch a screenshot
            adb_command = ['adb', 'exec-out', 'screencap', '-p']
            adb_process = subprocess.Popen(adb_command, stdout=subprocess.PIPE)

            # Read the raw image data
            raw_image_data = adb_process.stdout.read()

            # Convert the raw image data to a numpy array
            image_array = np.frombuffer(raw_image_data, dtype=np.uint8)

            # Decode the image to OpenCV format
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            return image
        except Exception as e:
            print(f"ADB fetch failed: {e}")
            return None

    def get_time_since_last_frame(self):
        """Calculate time since the last frame was fetched."""
        return time.time() - self.last_frame_time

    def stop(self):
        self.running = False
        if hasattr(self, 'capture'):
            self.capture.release()


# Independent testing of VideoFrameFetcher
def main():
    output_queue = queue.Queue(maxsize=1)
    video_source = "adb"  # You can change this to test adb input

    # Example usage with IP and port for ADB
    adb_ip = "192.168..242"  # Replace with your device's IP address
    adb_port = "5555"  # Optional, use default if not specified

    fetcher = VideoFrameFetcher(video_source, output_queue, adb_ip=adb_ip, adb_port=adb_port)
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
