import cv2
import numpy as np
import matplotlib.pyplot as plt

from tag_detector import *
from tag_processor import *
from homography_utils import *

if __name__ == "__main__":
    vid_cap = cv2.VideoCapture("assets/tagvideo.mp4")
    detector = TagDetector()
    tag_processor = TagProcessor()

    desired_tag_img_size = 200
    desired_corners = [(0,0), (desired_tag_img_size,0), (0,desired_tag_img_size), (desired_tag_img_size, desired_tag_img_size)]

    # cv2 tesxt params
    font = cv2.FONT_HERSHEY_SIMPLEX

    frame_counter = 0
    while True:
        _, frame = vid_cap.read()
        img_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frame_counter % 5 == 0:

            detector.set_tag_image(img_grayscale)
            tag_corners = detector.detect_tag_corners()  

            

            if len(tag_corners) == 4:  
                H_matrix = compute_tag_corners_homography_matrix(tag_corners, desired_corners)
                try:
                    isolated_tag_image = inverse_warp_image(H_matrix, img_grayscale, (desired_tag_img_size, desired_tag_img_size))
                    _, isolated_tag_image = cv2.threshold(isolated_tag_image, 127, 255, cv2.THRESH_BINARY) #performing thrsholding    

                    # processing the tag image
                    tag_processor.set_tag_image(isolated_tag_image)
                    tag_id, tag_orientaion = tag_processor.decode_tag()               

                    cv2.putText(frame, "Tag ID: "+str(tag_id)+" Orientation: "+str(tag_orientaion), (100, 100), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
                except:
                    pass

            for corner in tag_corners:
                cv2.circle(frame, (int(corner[0]), int(corner[1])), 3, (255, 0, 0), 5)


            cv2.imshow("canny_edge", frame)

            if cv2.waitKey(20) == ord("q"):
                break
        
        frame_counter+= 1

    vid_cap.release()
    cv2.destroyAllWindows()