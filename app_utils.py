import time
import cv2

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