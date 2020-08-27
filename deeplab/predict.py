from deeplab.nets.deeplab import Deeplabv3
from PIL import Image
import numpy as np
import random
import copy
import os
import cv2 as cv

def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (0,0,0))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image,nw,nh
    
def find_bbox(mask):
    _, labels, stats, centroids = cv.connectedComponentsWithStats(mask.astype(np.uint8))
    stats = stats[stats[:,4].argsort()]
    return stats[:-1]
    



