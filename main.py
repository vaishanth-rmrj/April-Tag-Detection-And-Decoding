import cv2
import numpy as np
import matplotlib.pyplot as plt

from tag_detector import *
from tag_processor import *
from homography_utils import *


if __name__ == "__main__":
    img_color = cv2.imread("assets/image_5.png")
    img_color = cv2.cvtColor(img_color , cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # detecting the tag from the image
    detector = TagDetector()
    detector.set_tag_image(img_gray)
    tag_corners = detector.detect_tag_corners()    

    plt.figure(figsize=(50,50)) 
    plt.subplot(2, 2, 1), plt.imshow(img_gray, cmap='gray') 
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 2, 2), plt.imshow(detector.get_fft_masked(), cmap='gray') 
    plt.title('Masked FFT'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 2, 3), plt.imshow(detector.get_inverse_fft(), cmap='gray')
    plt.title('Inverse FFT'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 2, 4), plt.imshow(img_gray, cmap='gray') 
    plt.subplot(2, 2, 4), plt.scatter(np.array(tag_corners)[:, 0], np.array(tag_corners)[:, 1], color="red") 
    plt.title('Tag corners detected'), plt.xticks([]), plt.yticks([])
    plt.show()

    # isolating the tag
    desired_corners = [(0,0), (200,0), (0,200), (200, 200)]
    H_matrix = compute_tag_corners_homography_matrix(tag_corners, desired_corners)
    isolated_tag_image = inverse_warp_image(H_matrix, img_gray, (200, 200))
    _, isolated_tag_image = cv2.threshold(isolated_tag_image, 127, 255, cv2.THRESH_BINARY) #performing thrsholding    

    # processing the tag image
    tag_processor = TagProcessor()
    tag_processor.set_tag_image(isolated_tag_image)
    tag_id, tag_orientaion = tag_processor.decode_tag()


    plt.figure(figsize=(50,50)) 
    plt.subplot(2, 2, 1), plt.imshow(img_gray, cmap='gray') 
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 2), plt.imshow(isolated_tag_image, cmap='gray') 
    plt.title('Isolated tag Image - ID  '+ str(tag_id)+ '  Orientation  '+ str(tag_orientaion)), plt.xticks([]), plt.yticks([])
    plt.show()

    