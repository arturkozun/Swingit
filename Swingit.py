import cv2
import numpy as np
import time
import PoseEstimationModule as pem

##########
# Variables:
# The minimum acceptable angle under armpit of a dancer (below it, the drop count starts)
threshold = 70

##########

# Upload a video
cap = cv2.VideoCapture("PoseVideos/bev4.MOV")

# Capture a live video from a camera ("0" most likely indicates a default built-in camera)
# cap = cv2.VideoCapture(0)

# Instantiate a class from PoseEstimationModule.py
detector = pem.PoseDetector()

# Initiate parameters
# The previous time
p_time = 0
# Left and right arm drop count
l_count = 0
r_count = 0

# Get the width and the height of the image
img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_video = int(cap.get(cv2.CAP_PROP_FPS))
size = (img_w, img_h)

# Define name, codec, and other parameters of an output video (if you want to save the video)
# result =cv2.VideoWriter("Bev_processed.avi", cv2.VideoWriter_fourcc(*'mp4v'), fps_video, size)

# Create a loop to continuously read and display frames from the video
while True:
    success, img = cap.read()
    # Optional flip for the live camera (so it works like a mirror)
    # img = cv2.flip(img, 1)
    key = cv2.waitKey(1)
    # Pause video when 'p' key is pressed
    if key == ord('p'):
        while True:
            key = cv2.waitKey(1)
            if key == ord('p'):
                break
    # Exit when ESC key is pressed
    elif key == 27:
        break
    # Resize the video if needed
    # img = cv2.resize(img, (720, 1280))

    # Run the model to find landmarks and put them to the list, called lm_list
    img = detector.find_pose(img, False)
    lm_list = detector.find_position(img, False)
    # print(lm_list)

    # If landmarks have been detected
    if len(lm_list) != 0:
        # Camera looking at the front of a person
        if lm_list[24][1] < lm_list[23][1]:
            angle_left_arm = detector.find_angle(img, 13, 11, 23)
            angle_right_arm = detector.find_angle(img, 24, 12, 14)

        # Camera looking at the back of a person
        elif lm_list[24][1] > lm_list[23][1]:
            angle_left_arm = detector.find_angle(img, 23, 11, 13)
            angle_right_arm = detector.find_angle(img, 14, 12, 24)

        # Calculate the drop count for the left and right arm separately
        if angle_left_arm < threshold:
            l_count += 1
        if angle_right_arm < threshold:
            r_count += 1

        # Draw the left hand drop count
        cv2.rectangle(img, (20, 20), (220, 120), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f"L", (30, 110), cv2.FONT_HERSHEY_PLAIN, 2,
                    (255, 0, 0,), 5)
        cv2.putText(img, f"{int(l_count)}", (50, 90), cv2.FONT_HERSHEY_PLAIN, 5,
                    (255, 0, 0,), 5)

        # Draw the right hand drop count
        cv2.rectangle(img, (270, 20), (470, 120), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f"R", (280, 110), cv2.FONT_HERSHEY_PLAIN, 2,
                    (255, 0, 0,), 5)
        cv2.putText(img, f"{int(r_count)}", (300, 90), cv2.FONT_HERSHEY_PLAIN, 5,
                    (255, 0, 0,), 5)

    # Calculate and display frames per second
    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time
    cv2.putText(img, f"{int(fps)}", (900, 100), cv2.FONT_HERSHEY_PLAIN, 5,
                (255, 0, 0,), 5)

    # Show the image
    cv2.imshow("Image", img)
    cv2.waitKey(1)

    # Save the output video
    # result.write(img)

# Release all the frames
result.release()
cap.release()
cv2.destroyAllWindows()
