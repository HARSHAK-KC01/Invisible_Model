
import cv2
import time
import numpy as np

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Give the camera time to warm up
time.sleep(2)

# Capture the background (assuming no red object is present)
print("Capturing background... Please ensure no red object is in view.")
background_frames = []
for i in range(60):  # Capture more frames for a more stable background
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture background frame.")
        break
    background_frames.append(frame)

if not background_frames:
    print("Exiting: No background frames captured.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# Average the captured frames to get a stable background
background = np.mean(background_frames, axis=0).astype(np.uint8)
background = np.flip(background, axis=1) # Flip horizontally for mirror effect

print("Background captured. Starting invisible cloak effect.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame. Exiting...")
        break

    frame = np.flip(frame, axis=1)  # Flip horizontally for mirror effect

    # Convert frame to HSV color space for better color detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define range for red color in HSV
    # Lower red range
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    # Upper red range (due to hue wrapping around 180 degrees)
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # Create masks for red color
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    # Refine the mask using morphological operations
    # Remove small noise (erosion)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
    # Dilate to restore the size of the object and fill small holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8), iterations=1)

    # Create an inverse mask to get the non-red areas
    inverse_mask = cv2.bitwise_not(mask)

    # Segment the red color part from the background
    cloak_area = cv2.bitwise_and(background, background, mask=mask)
    # Segment the non-red part from the current frame
    current_area = cv2.bitwise_and(frame, frame, mask=inverse_mask)

    # Combine the cloak area and current area to create the final effect
    combined = cv2.add(cloak_area, current_area)

    # Display the resulting frame
    cv2.imshow("Invisible Cloak", combined)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
