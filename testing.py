import warnings
warnings.filterwarnings('ignore')
import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2
import random
import math
import re
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg

from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
from mrcnn.visualize import display_instances
import mrcnn.model as modellib
from mrcnn.model import log
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import utils

# Root directory of the project
'C:/Users/wuethral/Desktop/Project_nr_2'

ROOT_DIR = 'C:/Users/wuethral/Desktop/Project_nr_2/mrcnn'

DEFAULT_LOGS_DIR = 'C:/Users/wuethral/Desktop/Project_nr_2/trained_model'

MODEL_DIR = 'C:/Users/wuethral/Desktop/Project_nr_2/trained_model'

WEIGHTS_PATH = 'C:/Users/wuethral/Desktop/Project_nr_2/trained_model/object20210907T1523/mask_rcnn_object_0010.h5'

class CustomConfig(Config):
    """Configuration for training on the custom  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "object"

    IMAGES_PER_GPU = 1

    NUM_CLASSES = 2 + 1  # Background + phone,laptop and mobile

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 10

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

config = CustomConfig()
DIR = 'C:/Users/wuethral/Desktop/Project_nr_2/mrcnn'
CUSTOM_DIR = os.path.join(ROOT_DIR, "/DataSet/")
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.7

config = InferenceConfig()
config.display()

class CustomDataset(utils.Dataset):

    def load_custom(self, dataset_dir, subset):

        """Load a subset of the Dog-Cat dataset.
               dataset_dir: Root directory of the dataset.
               subset: Subset to load: train or val
               """
        # Add classes. We have only one class to add.
        self.add_class('object', 1, 'Car')
        self.add_class('object', 2, 'Motorcycle')

        # Train or validation dataset?
        assert subset in ['train', 'val']

        dataset_dir = os.path.join(dataset_dir, subset)
        PATH_TO_JSON = 'C:/Users/wuethral/Desktop/Project_nr_2/DataSet/' + subset + '/polygon_labels.json'
        label_info = json.load(open(PATH_TO_JSON))

        image_names = []

        for file_name in os.listdir(dataset_dir):

            old_file = os.path.join(dataset_dir, file_name)
            if old_file != dataset_dir + '\\.DS_Store':
                file_name = file_name.split('.')[0]
                #print(file_name)
                image_names.append(file_name)
        #print(image_names)
        label_info = list(label_info.values())
        #print(label_info)
        label_info = [i for i in label_info if i['filename'] in image_names]
        #print(label_info)

        for i in label_info:

            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            polygons = [s['shape_attributes'] for s in i['regions']]
            objects = [r['region_attributes']['label'] for r in i['regions']]
            print('objects:', objects)

            name_dict = {'Car': 1, 'Motorcycle': 2}

            num_ids = [name_dict[i] for i in objects]

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.

            print('numids', num_ids)
            print('file:', i)
            file_name = i['filename'] + '.jpg'
            image_path = os.path.join(dataset_dir, file_name)
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                'object',  ## for a single class just add the name here
                image_id=i['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                num_ids=num_ids,
            )

    def load_mask(self, image_id):

        """Generate instance masks for an image.
              Returns:
               masks: A bool array of shape [height, width, instance count] with
                   one mask per instance.
               class_ids: a 1D array of class IDs of the instance masks.
               """
        # If not a Dog-Cat dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        if info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)
        num_ids = info['num_ids']
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                            dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])

            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        # Map class names to class IDs.
        num_ids = np.array(num_ids, dtype=np.int32)
        return mask, num_ids  # np.ones([mask.shape[-1]], dtype=np.int32)

# Inspect the model in training or inference modes values: 'inference' or 'training'
TEST_MODE = "inference"
ROOT_DIR ='C:/Users/wuethral/Desktop/Project_nr_2/DataSet'
def get_ax(rows=1, cols=1, size=16):
  """Return a Matplotlib Axes array to be used in all visualizations in the notebook.  Provide a central point to control graph sizes. Adjust the size attribute to control how big to render images"""
  _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
  return ax

# Load validation dataset
# Must call before using the dataset
CUSTOM_DIR = 'C:/Users/wuethral/Desktop/Project_nr_2/DataSet'
dataset = CustomDataset()
dataset.load_custom(CUSTOM_DIR, "val")
dataset.prepare()
print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

config = CustomConfig()
#LOAD MODEL. Create model in inference mode
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load COCO weights Or, load the last model you trained
weights_path = WEIGHTS_PATH
# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)


#Now, we are ready for testing our model on any image.

#RUN DETECTION
image_id = random.choice(dataset.image_ids)
print("image id is :",image_id)
image, image_meta, gt_class_id, gt_bbox, gt_mask =\
modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
info = dataset.image_info[image_id]
print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,dataset.image_reference(image_id)))

# Run object detection
results = model.detect([image], verbose=1)

# Display results
x = get_ax(1)
r = results[0]
ax = plt.gca()
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], dataset.class_names, r['scores'], ax=ax, title="Predictions")


log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)

''' 
# This is for predicting images which are not present in dataset
path_to_new_image = 'C:/Users/wuethral/Desktop/Project_nr_2/testing_images/testcat.jpg'
image1 = mpimg.imread(path_to_new_image)

# Run object detection
print(len([image1]))
results1 = model.detect([image1], verbose=1)

# Display results
ax = get_ax(1)
r1 = results1[0]
visualize.display_instances(image1, r1['rois'], r1['masks'], r1['class_ids'],
dataset.class_names, r1['scores'], ax=ax, title="Predictions1")


path_to_new_image = 'C:/Users/wuethral/Desktop/Project_nr_2/testing_images/testcat.jpg'
image1 = mpimg.imread(path_to_new_image)

# Run object detection
print(len([image1]))
results1 = model.detect([image1], verbose=1)


# Display results
ax = get_ax(1)
r1 = results1[0]
visualize.display_instances(image1, r1['rois'], r1['masks'], r1['class_ids'],
dataset.class_names, r1['scores'], ax=ax, title="Predictions1")
'''