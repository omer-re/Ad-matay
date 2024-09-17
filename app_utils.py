import time
import cv2
import numpy as np
import subprocess
from functools import wraps


def time_logger(timing_dict_or_attr_name):
    """
    A flexible decorator to log the execution time of both regular functions and methods.

    If the decorated function is a class method, provide the name of the instance attribute
    (as a string) where timings should be stored. If it's a regular function, pass in a dictionary.

    Args:
        timing_dict_or_attr_name (dict or str):
            - For regular functions, pass a dictionary to log function timings.
            - For methods, pass the attribute name (string) where timing info is stored.

    Returns:
        callable: The decorated function with timing and logging functionality.

    Example for regular function:
        >>> timings = {}
        >>> @time_logger(timings)
        ... def example_function():
        ...     time.sleep(0.1)
        >>> example_function()
        >>> 'example_function' in timings
        True
        >>> timings['example_function'] >= 0.1
        True

    Example for class method:
        >>> class MyClass:
        ...     def __init__(self):
        ...         self.timing_info = {}
        ...
        ...     @time_logger('timing_info')
        ...     def method(self):
        ...         time.sleep(0.2)
        >>> obj = MyClass()
        >>> obj.method()
        >>> 'method' in obj.timing_info
        True
        >>> obj.timing_info['method'] >= 0.2
        True
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if isinstance(timing_dict_or_attr_name, dict):
                # Handle regular function case
                timing_dict = timing_dict_or_attr_name
                start_time = time.time()
                result = func(*args, **kwargs)
                elapsed_time = time.time() - start_time
                timing_dict[func.__name__] = elapsed_time
                return result
            elif isinstance(timing_dict_or_attr_name, str):
                # Handle method case (instance method or class method)
                instance = args[0]  # 'self' or 'cls'
                timing_dict = getattr(instance, timing_dict_or_attr_name, {})
                start_time = time.time()
                result = func(*args, **kwargs)
                elapsed_time = time.time() - start_time
                timing_dict[func.__name__] = elapsed_time
                setattr(instance, timing_dict_or_attr_name, timing_dict)  # Update instance timing info
                return result
            else:
                raise TypeError("Argument should be a dictionary or a string for instance attribute name.")

        return wrapper

    return decorator



def time_measurement(variable):
    """
    Decorator to measure execution time and store it in the given variable.

    :param variable: A mutable container (e.g., a dict, list, or an object with attributes)
                     to store the execution time of the decorated function.

    Example Usage:

    # Create a container to store the execution time
    timing_info = {}

    @time_measurement(timing_info)
    def my_function():
        # Simulate some work with time.sleep()
        time.sleep(2)
        return "Function completed"

    # Call the decorated function
    result = my_function()

    # Access the execution time
    print(f"Execution time: {timing_info['execution_time']} seconds")

    Output:
    Execution time: 2.00xx seconds
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)  # Call the decorated function
            end_time = time.time()
            execution_time = end_time - start_time  # Measure time
            variable['execution_time'] = execution_time  # Store execution time in the provided variable
            return result
        return wrapper
    return decorator


def add_timing_to_frame(timing_info, frame):

    duration=timing_info
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"Processing time", (int(0.6*w), h-100), font, 2, (200, 100, 200), 3,cv2.LINE_AA)
    cv2.putText(frame, f"{duration:.2f}s", (int(0.6*w), h-40), font, 2.5, (200, 100, 200), 3,cv2.LINE_AA)

    return frame


def create_black_frame(width, height):
    """
    Creates a blank (black) frame with the given width and height.
    """
    return np.zeros((height, width, 3), dtype=np.uint8)

def add_title(frame, title):
    """
    Adds a title to the top of the frame.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, title, (10, 30), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
    return frame



def resize_frame(frame, width, height):
    """
    Resizes the given frame to the specified width and height.
    """
    return cv2.resize(frame, (width, height))

def connect_adb_to_device(ip, port):
    """
    ADB attributes, currently disabled.

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

def fetch_adb_frame():
    """
            ADB attributes, currently disabled.

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