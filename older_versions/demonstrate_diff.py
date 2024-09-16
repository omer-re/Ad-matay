import cv2
import numpy as np

def main():
    # Open the USB camera (you may need to adjust the index if multiple cameras are connected)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    # List to hold the last 20 frames
    frames = []

    # Main loop to capture frames
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Convert the frame to grayscale for comparison
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Add the new frame to the list of frames
        frames.append(gray_frame)

        # Keep only the last 20 frames
        if len(frames) > 50:
            frames.pop(0)

        # If we have 20 frames, calculate the difference mask
        if len(frames) == 50:
            # Initialize an empty mask of the same size as the frames
            change_mask = np.zeros_like(frames[0])

            # Compare each frame to the previous one and accumulate changes
            for i in range(1, len(frames)):
                diff = cv2.absdiff(frames[i], frames[i - 1])
                # Mark pixels that changed in the mask
                change_mask = cv2.bitwise_or(change_mask, diff)

            # Threshold the mask to enhance changes (you can adjust the threshold value)
            _, change_mask = cv2.threshold(change_mask, 25, 255, cv2.THRESH_BINARY)

            # Display the mask of changes
            cv2.imshow("Change Mask", change_mask)

        # Display the live camera feed
        cv2.imshow("Camera Feed", frame)

        # Exit loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
