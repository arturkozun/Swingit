import cv2
import mediapipe as mp
import time
import math
import numpy as np

# Define the class to detect pose
class PoseDetector:
    # Define the variables of the object
    def __init__(self, mode=False, model_compl=1, smooth_lm=True, enable_segment=False,
                 smooth_segment=True, detection_conf=0.5, tracking_conf=0.5):
        self.mode = mode
        self.model_compl = model_compl
        self.smooth_lm = smooth_lm
        self.enable_segment = enable_segment
        self.smooth_segment = smooth_segment
        self.detection_conf = detection_conf
        self.tracking_conf = tracking_conf

        # Define the drawing utilities
        self.mp_draw = mp.solutions.drawing_utils
        # Enable pose estimation from mediapipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(self.mode, self.model_compl, self.smooth_lm,
                                      self.enable_segment, self.smooth_segment,
                                      self.detection_conf, self.tracking_conf)

    # Find a pose, using mediapipe (set draw=True if you want to display it)
    def find_pose(self, img, draw=True):

        # Convert the image from bgr to rgb (mediapipe library operates in rgb)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Send the image to the model
        self.results = self.pose.process(img_rgb)

        # If landmarks have been captured
        if self.results.pose_landmarks:
            if draw:
                # Draw the landmarks and the connections
                self.mp_draw.draw_landmarks(img, self.results.pose_landmarks,
                                            self.mp_pose.POSE_CONNECTIONS)
        return img

    # Find position of each landmark and save it into a list
    def find_position(self, img, draw=True):
        # Create an empty landmark list
        self.lm_list = []
        # If landmarks have been captured
        if self.results.pose_landmarks:
            # Access the id and the coordinates of each landmark
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                # Get the dimensions of the image
                h, w, c = img.shape
                # Convert the coordinates from decimal image ratio into the pixel value
                cx, cy = int(lm.x * w), int(lm.y * h)
                # Append the id and the coordinates to the landmark list
                self.lm_list.append([id, cx, cy])
                if draw:
                    # Draw extra circles on the landmarks
                    cv2.circle(img, (cx ,cy), 5, (245, 239, 34), cv2.FILLED)

        return self.lm_list

    # Convert the angle to color
    def angle_to_color(self, angle):
        if angle < 70:
            return (0, 0, 255)
        if angle >= 70:
            return (0, 255, 0)

    # Find the angle in a chosen joint (example: for the right shoulder set p2 to 12)
    def find_angle(self, img, p1, p2, p3, draw=True):

        # Get the landmarks
        x1, y1 = self.lm_list[p1][1:]
        x2, y2 = self.lm_list[p2][1:]
        x3, y3 = self.lm_list[p3][1:]

        # Calculate the angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                            math.atan2(y1 - y2, x1 - x2))

        # Convert the negative angles to positive
        if angle < 0:
            angle += 360

        # Change the color relative to the angle
        color = self.angle_to_color(angle)

        # Draw chosen landmarks and connections with changeable color
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), color, 3)
            cv2.line(img, (x3, y3), (x2, y2), color, 3)
            cv2.circle(img, (x1, y1), 2, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 4, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 2, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 4, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 2, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 4, (0, 0, 255), 2)
            cv2.putText(img, str(int(angle)), (x2 + 20, y2 + 20),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0 , 255), 2)
        return angle

# Main function (script to test the module)
def main():
    cap = cv2.VideoCapture("DanceVideos/bev4.MOV")
    p_time = 0
    detector = PoseDetector()
    while True:
        success, img = cap.read()
        img = detector.find_pose(img)
        lm_list = detector.find_position(img, draw=False)
        if len(lm_list) != 0:
            print(lm_list[14])
            # cv2.circle(img, (lm_list[14][1], lm_list[14][2]), 15, (0, 0, 255), cv2.FILLED)

        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    3, (255, 0, 13), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(10)


# If run this python file, execute main function. Otherwise just import the elements from the module
if __name__ == "__main__":
    main()