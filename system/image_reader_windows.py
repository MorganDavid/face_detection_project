import sys
import argparse
import cv2
import numpy as np
from run_one_image_new import image_predictor

def read_cam(cap):
    windowName = "Face Detection"

    cv2.namedWindow(windowName, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(windowName,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

    cv2.setWindowTitle(windowName,"Face Detection")
    showFullScreen = False

    predictor = image_predictor()
    while not cap.isOpened():
        cap.open()

    while True:
        # Capture frame-by-frame
        ret, displayBuf = cap.read()        

        old_height, old_width, _ = displayBuf.shape

        scale_factor, n_width, n_height = predictor.scale_image_to_face_size(displayBuf, 10, 50)
        norm_image = cv2.normalize(displayBuf, None, alpha=-1, beta=+1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        blah = predictor.detect_faces(norm_image,scale_factor)
        if blah[0]==-1: 
            print("ERROR")
            continue
        _, init_boxes, boxes = predictor.detect_faces(norm_image,scale_factor)
        image = cv2.resize(displayBuf,(int(old_width),int(old_height)))
        final_im = predictor.create_img_with_recs(boxes,displayBuf,int(old_width/n_width))
       # print(boxes)

        cv2.putText(final_im,'\'<\' and \'>\' to change threshold. Current: '+str(predictor.get_r_thresh()), 
        (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1)
        cv2.putText(final_im,('\'s\' to switch to {}, ESC to exit'.format('sliding window' if predictor.get_seg() else 'SSS')), 
        (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1)


        cv2.imshow(windowName,final_im)
        key=cv2.waitKey(10)
        if key == 27: # Check for ESC key
                # When everything done, release the capture
            cap.release()
            cv2.destroyAllWindows()
            break
        amnt = 0.01
        if key == 46:
            predictor.inc_r_thresh(amnt)
        if key == 44:
            predictor.dec_r_thresh(amnt)
        if key == 114:
            predictor.reset_r_thresh()
        if key == 115:
            predictor.toggle_seg()
    else:
     print ("camera open failed")

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    read_cam(cap)