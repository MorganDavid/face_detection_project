import sys
import argparse
import cv2
import numpy as np
from run_one_image import image_predictor

def parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_device", dest="video_device",
                        help="Video device # of USB webcam (/dev/video?) [0]",
                        default=0, type=int)
    arguments = parser.parse_args()
    return arguments

# On versions of L4T previous to L4T 28.1, flip-method=2
# Use the Jetson onboard camera
def open_onboard_camera():
    return cv2.VideoCapture("nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480, format=(string)I420, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")

# Open an external usb camera /dev/videoX
def open_camera_device(device_number):
    return cv2.VideoCapture(device_number)
   

def read_cam(video_capture):
    if video_capture.isOpened():
        windowName = "Face Detection"
        cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(windowName,250,200)
        cv2.moveWindow(windowName,0,0)
        cv2.setWindowTitle(windowName,"Face Detection")
        showFullScreen = False
        predictor = image_predictor()
        while True:
            if cv2.getWindowProperty(windowName, 0) < 0: # Check to see if the user closed the window
                # This will fail if the user closed the window; Nasties get printed to the console
                break
            ret_val, frame = video_capture.read()
            
            displayBuf = frame #displayBuf is the CV2 image object
            

            old_height, old_width, _ = displayBuf.shape

            scale_factor, n_width, n_height = predictor.scale_image_to_face_size(displayBuf, 10, 20)
	
            init_boxes, boxes = predictor.detect_faces(displayBuf,scale_factor)
            image = cv2.resize(displayBuf,(int(old_width),int(old_height)))
            final_im = predictor.create_img_with_recs(boxes,displayBuf,int(old_width/n_width))

            cv2.imshow(windowName,final_im)
            key=cv2.waitKey(10)
            if key == 27: # Check for ESC key
                cv2.destroyAllWindows()
                break 
              
    else:
     print ("camera open failed")



if __name__ == '__main__':
    arguments = parse_cli_args()
    print("Called with args:")
    print(arguments)
    print("OpenCV version: {}".format(cv2.__version__))
    print("Device Number:",arguments.video_device)
    if arguments.video_device==0:
      video_capture=open_onboard_camera()
    else:
      video_capture=open_camera_device(arguments.video_device)
    read_cam(video_capture)
    video_capture.release()
    cv2.destroyAllWindows()
