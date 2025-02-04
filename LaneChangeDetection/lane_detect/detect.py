import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from nav_msgs.msg import Odometry

import cv2
import time
import os
import argparse
import numpy as np
from .dnn_detect import Dnn

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import torch
import torchvision
import random
import math
import sys
#import pillow

#from PIL import Image # check alternative
from torchvision.transforms import ToTensor, ToPILImage
from matplotlib.pyplot import imshow, figure, subplots

#Model loading
from .erfnet import Net as ERFNet
from .lcnet import Net as LCNet

# utils
from .functions import color_lanes, blend
    
# Descriptor size definition
DESCRIPTOR_SIZE = 64

# Maximum number of lanes the network has been trained with + background
NUM_CLASSES_SEGMENTATION = 5

# Maxmimum number of classes for classification
NUM_CLASSES_CLASSIFICATION = 3

# Image size
HEIGHT = 360
WIDTH = 640

#nms = 0.4

class Detect(Node):
	def __init__(self,topic_name='text_message',nms=0.4):#here     
		super().__init__('detect')        

		# <...> Create a subscription:

		self.subscription = self.create_subscription(Image,topic_name,self.callback,1)# change '/cam_front/raw' 
		self.subscription

		self.get_logger().info(f'Subscribed to: {topic_name}')
		self.dnn_img = Dnn(nms)

		self.DESCRIPTOR_SIZE = 64
		self.NUM_CLASSES_SEGMENTATION = 5
		self.NUM_CLASSES_CLASSIFICATION = 3
		self.HEIGHT = 360
		self.WIDTH = 640
		# to cuda or not to cuda
		if torch.cuda.is_available():
		    self.map_location=lambda storage, loc: storage.cuda()
		else:
		    self.map_location='cpu'

		self.caution_flag = 0
		
		

	def callback(self, img_data):


		image_message = img_data
		bridge = CvBridge()
		cv_image = bridge.imgmsg_to_cv2(image_message, 'bgr8')
		cv2.waitKey(1)

		#im = cv_image

		frame = cv_image

		classes, scores, boxes = self.dnn_img.detect(frame)
		im = self.dnn_img.draw(frame, classes, scores, boxes,self.caution_flag)

		dsize = (self.WIDTH, self.HEIGHT)
		output = cv2.resize(im, dsize)
		im = output

		im_tensor = ToTensor()(im)
		im_tensor = im_tensor.unsqueeze(0)

		# Creating CNNs and loading pretrained models
		segmentation_network = ERFNet(self.NUM_CLASSES_SEGMENTATION)
		classification_network = LCNet(self.NUM_CLASSES_CLASSIFICATION, self.DESCRIPTOR_SIZE, self.DESCRIPTOR_SIZE)


		segmentation_network.load_state_dict(torch.load('/mnt/home/desaisar/av/ros_ws/src/AV_Project_11/lab_project/lane_detect/lane_detect/erfnet_tusimple.pth', map_location = self.map_location))
		
		model_path = '/mnt/home/desaisar/av/ros_ws/src/AV_Project_11/lab_project/lane_detect/lane_detect/classification_{}_{}class.pth'.format(self.DESCRIPTOR_SIZE, self.NUM_CLASSES_CLASSIFICATION)
		classification_network.load_state_dict(torch.load(model_path, map_location = self.map_location))

		segmentation_network = segmentation_network.eval()
		classification_network = classification_network.eval()

		if torch.cuda.is_available():
		    segmentation_network = segmentation_network.cuda()
		    classification_network = classification_network.cuda()
    
		# Inference on instance segmentation
		if torch.cuda.is_available():
		    im_tensor = im_tensor.cuda()

		out_segmentation = segmentation_network(im_tensor)
		out_segmentation = out_segmentation.max(dim=1)[1]

		# Converting to numpy for visualization
		out_segmentation_np = out_segmentation.cpu().numpy()[0]
		out_segmentation_viz = np.zeros((self.HEIGHT, self.WIDTH, 3))

		
		for i in range(1, self.NUM_CLASSES_SEGMENTATION):
		    rand_c1 = random.randint(1, 255)
		    rand_c2 = random.randint(1, 255)    
		    rand_c3 = random.randint(1, 255)
		    out_segmentation_viz = color_lanes(
		        out_segmentation_viz, out_segmentation_np, 
		        i, (rand_c1, rand_c2, rand_c3), self.HEIGHT, self.WIDTH)
		

		#im_pil = Image.fromarray(im)
		im_pil = im
		im_seg_pil = blend(im_pil, out_segmentation_viz)

		im_seg = np.asarray(im_seg_pil)


		cv2.imshow('detected lanes uncalssified',im_seg)
		#cv2.waitKey(0)

		descriptors, index_map = extract_descriptors(out_segmentation, im_tensor)
  
		
		# Inference on descriptors
		classes = classification_network(descriptors).max(1)[1]

		# Class visualization
		out_classification_viz = np.zeros((self.HEIGHT, self.WIDTH, 3))

		num_lanes = len(index_map)
		if (num_lanes % 2) == 0:
		    mid_index_1 = int(num_lanes/2) - 1
		    mid_index_2 = int(num_lanes/2)
		    #is_even = 1
		else:
		    mid_index_1 = math.floor(num_lanes/2)
		    mid_index_2 = mid_index_1
		    #is_even = 0

		for i, lane_index in index_map.items():
		    if classes[i] == 0: # Continuous
		        color = (255, 0, 0)
		    elif classes[i] == 1: # Dashed
		        color = (0, 255, 0)
		    elif classes[i] == 2: # Double-dashed
		        color = (0, 0, 255)
		    else:
		        raise

		    if (i == mid_index_1) or (i == mid_index_2) :
		        if (classes[i] == 0) or (classes[i] == 2):
		            self.caution_flag = 1
		        else:
		            self.caution_flag = 0

		    out_classification_viz[out_segmentation_np == lane_index] = color

		#cv2.imshow(out_classification_viz.astype(np.uint8))

		#print(out_classification_viz.astype(np.uint8))# tensor type

		class_out_img = out_classification_viz.astype(np.uint8)
		out_img = np.asarray(class_out_img)

		cv2.imshow('detected lanes',out_img)

		#frame = out_img

		#classes, scores, boxes = self.dnn_img.detect(frame)
		#im = self.dnn_img.draw(frame, classes, scores, boxes)


		#cv2.waitKey(0)
		#cv2.destroyAllWindows()

		return

def extract_descriptors(label, image):
	# avoids problems in the sampling
	eps = 0.01

	# Descriptor size definition
	DESCRIPTOR_SIZE = 64

	# Maximum number of lanes the network has been trained with + background
	NUM_CLASSES_SEGMENTATION = 5

	# Maxmimum number of classes for classification
	NUM_CLASSES_CLASSIFICATION = 3

	# Image size
	HEIGHT = 360
	WIDTH = 640

	# The labels indices are not contiguous e.g. we can have index 1, 2, and 4 in an image
	# For this reason, we should construct the descriptor array sequentially
	inputs = torch.zeros(0, 3, DESCRIPTOR_SIZE, DESCRIPTOR_SIZE)
	if torch.cuda.is_available():
	    inputs = inputs.cuda()

	# This is needed to keep track of the lane we are classifying
	mapper = {}
	classifier_index = 0

	# Iterating over all the possible lanes ids
	for i in range(1, NUM_CLASSES_SEGMENTATION):
	    # This extracts all the points belonging to a lane with id = i
	    single_lane = label.eq(i).view(-1).nonzero().squeeze()#there was nothing inside nonzero()

	    # As they could be not continuous, skip the ones that have no points
	    if single_lane.numel() == 0 or len(single_lane.size()) == 0:
	        continue

	    # Points to sample to fill a squared desciptor
	    sample = torch.zeros(DESCRIPTOR_SIZE * DESCRIPTOR_SIZE)
	    if torch.cuda.is_available():
	        sample = sample.cuda()

	    sample = sample.uniform_(0, single_lane.size()[0] - eps).long()
	    sample, _ = sample.sort()

	    # These are the points of the lane to select
	    points = torch.index_select(single_lane, 0, sample)

	    # First, we view the image as a set of ordered points
	    descriptor = image.squeeze().view(3, -1)

	    # Then we select only the previously extracted values
	    descriptor = torch.index_select(descriptor, 1, points)

	    # Reshape to get the actual image
	    descriptor = descriptor.view(3, DESCRIPTOR_SIZE, DESCRIPTOR_SIZE)
	    descriptor = descriptor.unsqueeze(0)

	    # Concatenate to get a batch that can be processed from the other network
	    inputs = torch.cat((inputs, descriptor), 0)

	    # Track the indices
	    mapper[classifier_index] = i
	    classifier_index += 1

	return inputs, mapper




def main(args=None):
	rclpy.init(args=args)

	parser = argparse.ArgumentParser(description='Image type arguments')
	parser.add_argument('--nms',      default=0.4,    dest='nms',     type=float, help="add int")
	parser.add_argument('topic_name',      default='/cam_front/raw',     type=str, help="Image topic to subscribe to")

	args, unknown = parser.parse_known_args() 
	if unknown: print('Unknown args:',unknown)

	nms = args.nms
	topic_name = args.topic_name

	dtct = Detect(topic_name=topic_name,nms=nms)
	rclpy.spin(dtct)

