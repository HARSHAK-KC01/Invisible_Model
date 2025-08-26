
import cv2
import time
import numpy as np

# Global variables for color selection and tolerance
selected_hsv_color = None
color_range_set = False
hue_tolerance = 10
saturation_tolerance = 50
value_tolerance = 50

# Mouse callback function to select color
def select_color(event, x, y, flags, param):
    global selected_hsv_color, color_range_set
    if event == cv2.EVENT_LBUTTONDOWN:
        # Get the HSV value of the clicked pixel from the passed parameter
        hsv_frame = param[0]
        if hsv_frame is not None:
            selected_hsv_color = hsv_frame[y, x]
            color_range_set = True
            print(f"Selected HSV color: {selected_hsv_color}")

# Callback function for trackbars (does nothing, but required)
def set_hue_tolerance(val):
    global hue_tolerance
    hue_tolerance = val

def set_saturation_tolerance(val):
    global saturation_tolerance
    saturation_tolerance = val

def set_value_tolerance(val):
    global value_tolerance
    value_tolerance = val

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Give the camera time to warm up
time.sleep(2)

# Create background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

print("Capturing background... Please ensure no object to be cloaked is in view.")
# Capture initial frames to train the background subtractor
for _ in range(60):
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture background frame.")
        break
    frame = np.flip(frame, axis=1) # Flip horizontally for mirror effect
    fgbg.apply(frame, learningRate=0.01) # Apply with a low learning rate

print("Background initialized. Starting invisible cloak effect.")
print("Click on the color you want to make invisible in the 'Original Frame' window.")

# Create windows and trackbars
cv2.namedWindow("Original Frame")
hsv_frame_container = [None] # A mutable list to hold the current hsv frame
cv2.setMouseCallback("Original Frame", select_color, param=hsv_frame_container) # Set callback once

cv2.namedWindow("Settings")
cv2.createTrackbar("Hue Tolerance", "Settings", hue_tolerance, 50, set_hue_tolerance)
cv2.createTrackbar("Saturation Tolerance", "Settings", saturation_tolerance, 255, set_saturation_tolerance)
cv2.createTrackbar("Value Tolerance", "Settings", value_tolerance, 255, set_value_tolerance)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame. Exiting...")
        break

    frame = np.flip(frame, axis=1)  # Flip horizontally for mirror effect
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # Get the current hsv frame
    hsv_frame_container[0] = hsv # Update the hsv frame in the container for the callback

    cv2.imshow("Original Frame", frame)

    # Apply background subtraction
    fgmask = fgbg.apply(frame, learningRate=0) # Apply with 0 learning rate to get current foreground mask
    background = fgbg.getBackgroundImage() # Get the current background image

    mask = np.zeros(frame.shape[:2], dtype=np.uint8) # Initialize an empty mask

    if color_range_set and selected_hsv_color is not None:
        lower_bound = np.array([max(0, selected_hsv_color[0] - hue_tolerance),
                                max(0, selected_hsv_color[1] - saturation_tolerance),
                                max(0, selected_hsv_color[2] - value_tolerance)])
        upper_bound = np.array([min(179, selected_hsv_color[0] + hue_tolerance),
                                min(255, selected_hsv_color[1] + saturation_tolerance),
                                min(255, selected_hsv_color[2] + value_tolerance)])

        # Handle hue wrapping for lower bound
        if selected_hsv_color[0] - hue_tolerance < 0:
            lower_bound1 = np.array([0, lower_bound[1], lower_bound[2]])
            upper_bound1 = np.array([selected_hsv_color[0] + hue_tolerance, upper_bound[1], upper_bound[2]])
            mask1 = cv2.inRange(hsv, lower_bound1, upper_bound1)

            lower_bound2 = np.array([180 + (selected_hsv_color[0] - hue_tolerance), lower_bound[1], lower_bound[2]])
            upper_bound2 = np.array([179, upper_bound[1], upper_bound[2]])
            mask2 = cv2.inRange(hsv, lower_bound2, upper_bound2)
            mask = mask1 + mask2
        # Handle hue wrapping for upper bound
        elif selected_hsv_color[0] + hue_tolerance > 179:
            lower_bound1 = np.array([selected_hsv_color[0] - hue_tolerance, lower_bound[1], lower_bound[2]])
            upper_bound1 = np.array([179, upper_bound[1], upper_bound[2]])
            mask1 = cv2.inRange(hsv, lower_bound1, upper_bound1)

            lower_bound2 = np.array([0, lower_bound[1], lower_bound[2]])
            upper_bound2 = np.array([(selected_hsv_color[0] + hue_tolerance) - 180, upper_bound[1], upper_bound[2]])
            mask2 = cv2.inRange(hsv, lower_bound2, upper_bound2)
            mask = mask1 + mask2
        else:
            mask = cv2.inRange(hsv, lower_bound, upper_bound)

        # Combine color mask with foreground mask to ensure only moving objects of the selected color are cloaked
        mask = cv2.bitwise_and(mask, fgmask)

        # Refine the mask using morphological operations
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)
        mask = cv2.GaussianBlur(mask, (5,5), 0) # Add Gaussian blur for smoother edges

    # Create an inverse mask to get the non-cloaked areas
    inverse_mask = cv2.bitwise_not(mask)

    # Segment the cloaked color part from the background
    cloak_area = cv2.bitwise_and(background, background, mask=mask)
    # Segment the non-cloaked part from the current frame
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
