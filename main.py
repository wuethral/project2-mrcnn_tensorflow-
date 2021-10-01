import os
import sys
import json
import numpy as np
import skimage.draw
import cv2
import mrcnn.visualize
import matplotlib.pyplot as plt

DIR_TO_MRCNN = 'C:/Users/wuethral/Desktop/Project_nr_2/mrcnn'
#print(DIR_TO_MRCNN)
#sys.path.append(DIR_TO_MRCNN)

# Import Mask RCNN

import mrcnn.config as config
import mrcnn.model as modellib
import mrcnn.utils as utils

ROOT_DIR = 'C:/Users/wuethral/Desktop/Project_nr_2'

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, 'mask-rcnn-coco.h5')

# Directory to save logs and model checkpoints
TRAINED_MODEL_DIR = os.path.join(ROOT_DIR, 'trained_model')

class CustomConfig(config.Config):

    """Configuration for training on the custom  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = 'object'

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 2 + 1

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 10

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

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

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "object":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

def train(model):

    dataset_train = CustomDataset()
    dataset_train.load_custom('C:/Users/wuethral/Desktop/Project_nr_2/DataSet', 'train')
    dataset_train.prepare()

    dataset_val = CustomDataset()
    dataset_val.load_custom('C:/Users/wuethral/Desktop/Project_nr_2/DataSet', 'val')
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.

    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=10,
                layers='heads')

config = CustomConfig()
model = modellib.MaskRCNN(mode='training', config=config,
                          model_dir=TRAINED_MODEL_DIR)

weights_path = COCO_WEIGHTS_PATH

if not os.path.exists(weights_path):
    utils.download_trained_weights(weights_path)

model.load_weights(weights_path, by_name=True, exclude=['mrcnn_class_logits', 'mrcnn_bbox_fc',
                                                        'mrcnn_bbox', 'mrcnn_mask'])

train(model)
