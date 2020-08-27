# -*- coding: utf-8 -*-
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2
import time
from mask.mrcnn.config import Config
from datetime import datetime
from PIL import Image, ImageFont, ImageDraw
from mask.mrcnn import utils
import mask.mrcnn.model as modellib



class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"
 
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
 
    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # background + 3 shapes
 
    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 448
    IMAGE_MAX_DIM = 448
 
    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8 * 6, 16 * 6, 32 * 6, 64 * 6, 128 * 6)  # anchor side in pixels
 
    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE =100
 
    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100
 
    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50
 
#import train_tongue
#class InferenceConfig(coco.CocoConfig):
class InferenceConfig(ShapesConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1



def detect(file):
        MODEL_DIR = "mask/logs"
        COCO_MODEL_PATH = "mask/mask_rcnn_shapes_0010.h5"
        config = InferenceConfig()
 
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
 
        # Create model object in inference mode.
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
 
        # Load weights trained on MS-COCO
        model.load_weights(COCO_MODEL_PATH, by_name=True)
 
        class_names = ['BG', 'fire','smoke']
        im = skimage.io.imread(str(file))
        image = Image.open(file)
        # Run detection
        results = model.detect([im], verbose=1)
        r = results[0]
        # Visualize results
        #visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],class_names, r['scores'])
        
        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(class_names), 1., 1.)
                      for x in range(len(class_names))]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),colors))
        
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.         
        
        font = ImageFont.truetype(font='arial.ttf',size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300
      
        for i in range(len(r['rois'])):
            class_id = r['class_ids'][i]
            predicted_class = class_names[class_id]
            box = r['rois'][i]
            score = r['scores'][i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))
            

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for j in range(thickness):
                draw.rectangle(
                    [left + j, top + j, right - j, bottom - j],
                    outline=colors[class_id])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=colors[class_id])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw


        if  len(r['rois']) > 1:    
            text = 'Found {} boxes for {} \n'.format(len(r['rois']), 'image')
        else:
            text = 'Found {} box for {} \n'.format(len(r['rois']), 'image')
        for i in range(len(r['rois'])):
            class_id = r['class_ids'][i]
            score = r['scores'][i] if r['scores'] is not None else None
            label = class_names[class_id]
            text += 'Box'+str(i+1)+':'+label +'   score='+str(round(score,3))+'\n'
        
        if 'fire' in text and 'smoke' in text:
            a = 'Fire & Smoke\n'
        elif 'fire' in text:
            a = 'Fire\n'
        elif 'smoke' in text:
            a = 'Smoke\n'
        else:
            a = 'Normal\n'
        text = 'Classification:'+ a + text
        
        return image,text

 