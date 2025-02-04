import torch
import torchvision
import numpy as np
import random
import math
import cv2

# Data loading and visualization imports
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from matplotlib.pyplot import imshow, figure, subplots

# Model loading
from models.erfnet import Net as ERFNet
from models.lcnet import Net as LCNet

# utils
from functions import color_lanes, blend

# to cuda or not to cuda
if torch.cuda.is_available():
    map_location=lambda storage, loc: storage.cuda()
else:
    map_location='cpu'
    
# Descriptor size definition
DESCRIPTOR_SIZE = 64

# Maximum number of lanes the network has been trained with + background
NUM_CLASSES_SEGMENTATION = 5

# Maxmimum number of classes for classification
NUM_CLASSES_CLASSIFICATION = 3

# Image size
HEIGHT = 360
WIDTH = 640

####################

def extract_descriptors(label, image):
    # avoids problems in the sampling
    eps = 0.01
    
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


####################

#im = Image.open('images/test.jpg')##########################################changed
# ipynb visualization
#%matplotlib inline ####################### check
#imshow(np.asarray(im))

im = cv2.imread('images/test.jpg')
cv2.imshow('window_name', im)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#im = im.resize((WIDTH, HEIGHT))

#############################################################
# dsize
dsize = (WIDTH, HEIGHT)

# resize image
output = cv2.resize(im, dsize)
im = output
##########################################################

im_tensor = ToTensor()(im)
im_tensor = im_tensor.unsqueeze(0)

# Creating CNNs and loading pretrained models
segmentation_network = ERFNet(NUM_CLASSES_SEGMENTATION)
classification_network = LCNet(NUM_CLASSES_CLASSIFICATION, DESCRIPTOR_SIZE, DESCRIPTOR_SIZE)

segmentation_network.load_state_dict(torch.load('pretrained/erfnet_tusimple.pth', map_location = map_location))
model_path = 'pretrained/classification_{}_{}class.pth'.format(DESCRIPTOR_SIZE, NUM_CLASSES_CLASSIFICATION)
classification_network.load_state_dict(torch.load(model_path, map_location = map_location))

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
out_segmentation_viz = np.zeros((HEIGHT, WIDTH, 3))

for i in range(1, NUM_CLASSES_SEGMENTATION):
    rand_c1 = random.randint(1, 255)
    rand_c2 = random.randint(1, 255)    
    rand_c3 = random.randint(1, 255)
    out_segmentation_viz = color_lanes(
        out_segmentation_viz, out_segmentation_np, 
        i, (rand_c1, rand_c2, rand_c3), HEIGHT, WIDTH)

#print(im)
im_pil = Image.fromarray(im)

im_seg_pil = blend(im_pil, out_segmentation_viz)
#print(im_seg_pil)

im_seg = np.asarray(im_seg_pil)
#print(im_seg)

cv2.imshow('window_name2',im_seg)################################################

#For cv2 to pil
#im_pil = Image.fromarray(img)

# For reversing the operation:
#im_np = np.asarray(im_pil)

##################

descriptors, index_map = extract_descriptors(out_segmentation, im_tensor)

GRID_SIZE = 2
_, fig = subplots(GRID_SIZE, GRID_SIZE)

for i in range(0, descriptors.size(0)):
    desc = descriptors[i].cpu()

    desc = ToPILImage()(desc)
    row = math.floor((i / GRID_SIZE))
    col = i % GRID_SIZE

    fig[row, col].imshow(np.asarray(desc))
    
# Inference on descriptors
classes = classification_network(descriptors).max(1)[1]
print(index_map)
print(classes)

# Class visualization
out_classification_viz = np.zeros((HEIGHT, WIDTH, 3))

for i, lane_index in index_map.items():
    if classes[i] == 0: # Continuous
        color = (255, 0, 0)
    elif classes[i] == 1: # Dashed
        color = (0, 255, 0)
    elif classes[i] == 2: # Double-dashed
        color = (0, 0, 255)
    else:
        raise
    out_classification_viz[out_segmentation_np == lane_index] = color

#cv2.imshow(out_classification_viz.astype(np.uint8))

#print(out_classification_viz.astype(np.uint8))# tensor type

class_out_img = out_classification_viz.astype(np.uint8)
out_img = np.asarray(class_out_img)

cv2.imshow('window_name3',out_img)
cv2.waitKey(0)
cv2.destroyAllWindows()



#print(im)
#im_pil = Image.fromarray(im)

#im_seg = np.asarray(im_seg_pil)
#print(im_seg)

#cv2.imshow('window_name2',im_seg)################################################



