import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN, PNet

detector = MTCNN()
pnet = detector.get_PNet()

image = cv2.imread("trump.jpg")
#Makes multiple scales of the input image. 
def make_scale_pyramid(img):
	width, height, depth = img.shape
	size_dec = 0.7 #scale to reduce by on every iteration
	num_of_imgs = 6 # number of images in the pyramid. To get the smallest image do: imgsize/(size_dec*num_of_imgs)
	im_pyramid = []
	for i in range(0,num_of_imgs):
		if i != 0:
			width = int(width*size_dec)
			height = int(height*size_dec)
			scaled_im = cv2.resize(img,(width,height), interpolation=cv2.INTER_AREA)
		else: 
			scaled_im = img
		scaled_im_norm = (scaled_im-127.5)*0.0078125
		im_pyramid.append(scaled_im_norm)
		print("our shape is: ",scaled_im_norm.shape)
	return im_pyramid
	
def face_detection_theirs(image):
	faces = detector.detect_faces(image)
	print(faces)
	for i in faces:
		box = i['box']
		nose = i['keypoints']['nose']
		print(box)
		cv2.rectangle(image,(box[0],box[1]),(box[0]+box[2],box[1]+box[3]),(0,255,0),3)
		cv2.circle(image, nose, 3, (255,0,0),3)
	cv2.imshow("a", image)
	cv2.waitKey()
face_detection_theirs(image)


def step1():
	image_pyramid = make_scale_pyramid(image)
	for img in image_pyramid:
		img1 = np.expand_dims(img, 0)
		img2 = np.transpose(img1, (0, 2, 1, 3))
		
		out = pnet.feed(img2)

		out0 = np.transpose(out[0], (0, 2, 1, 3))
		out1 = np.transpose(out[1], (0, 2, 1, 3))

		boxes, _ = detector.generate_bounding_box(out1[0, :, :, 1].copy(),out0[0, :, :, :].copy(), 1, [0.6,0.7,0.7])#The last parameter is the steps_threashold. Don't know what it does. 
		print(boxes)

step1()

