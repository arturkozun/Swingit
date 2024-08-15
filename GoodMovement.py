import cv2
import time
import numpy as np
import PoseEstimationModule as pem

#################
# Variables
video_path = "TennisVideos/carlos2.mp4"
point_to_draw = 16
brush_thickness = 5
draw_color = (0, 0, 255)

#################

# Upload a video
cap = cv2.VideoCapture(video_path)

# Instantiate a class from PoseEstimationModule.py
detector = pem.PoseDetector()

# Initiate parameters
# Previous time
p_time = 0
# Previous point detected
xp, yp = 0, 0

# Get the height and the width of the image
img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_video = int(cap.get(cv2.CAP_PROP_FPS))
size = (img_w, img_h)

# Create an extra canvas to draw the line on
img_canvas = np.zeros((img_h, img_w ,3), np.uint8)

# Define name, codec, and other parameters of an output video
# result =cv2.VideoWriter("tennis.avi", cv2.VideoWriter_fourcc(*'mp4v'), fps_video, size)

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

    # Coordinates of the current point
    xc, yc = lm_list[point_to_draw][1:]

    # Calculate and display frames per second
    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time
    cv2.putText(img, f"{int(fps)}", (10, 60), cv2.FONT_HERSHEY_PLAIN, 5,
                (255, 0, 0,), 5)



    # Draw the dot only for the first point
    if xp == 0 and yp == 0:
        xp, yp = xc, yc

    # Draw the line between the previous point and the current point
    cv2.line(img, (xp, yp), (xc, yc), draw_color, brush_thickness)

    cv2.line(img_canvas, (xp, yp), (xc, yc), draw_color, brush_thickness)

    # Set the current point as the previous point for the next iteration
    xp, yp = xc, yc

    # Convert the canvas into a grey image
    img_gray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
    # Convert the grey image into a binary inverse image
    _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    # Convert back to BGR
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
    # Merge original image with inverse image
    img = cv2.bitwise_and(img, img_inv)
    # Overlay the image with the canvas
    img = cv2.bitwise_or(img, img_canvas)


    # Show the processed video and/or the canvas
    cv2.imshow("Image", img)
    # cv2.imshow("Canvas", img_canvas)
    # Set the delay in miliseconds
    cv2.waitKey(1)

    # Save the output video
    # result.write(img)

# Release all the frames
result.release()
cap.release()
cv2.destroyAllWindows()