import os
import matplotlib.image as mpimg
import numpy as np
from PIL import Image,ImageOps
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms as transforms
import argparse
import random
import skimage
from skimage import transform
# flip
def H_flip(img):
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

def V_flip(img):
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    return img

# rotate
def RandomRotate(img_name):
    img= mpimg.imread(img_name)
    random.seed(230)
    angle = random.randint(-90,90)
    img = skimage.transform.rotate(img, angle, resize=False, center=(100, 100), mode='reflect', preserve_range = True)

    return img

#scale
def scale(img,scale = 0.5):
    x=400
    y=400
    y_s = y * scale
    x_s = x * scale
    out = img.resize((int(x_s),int(y_s)),Image.ANTIALIAS) #resize image with high-quality
    im1 = out
    im_h_np = np.hstack([im1, im1])
    im_v_np = np.vstack([im_h_np, im_h_np])
    return im_v_np


def zoom_in(img):

    y_s =  300
    x_s =  300
    out = img.resize((int(x_s),int(y_s)),Image.ANTIALIAS) #resize image with high-quality
    out=np.asarray(out)
    out1= cv2.hconcat([out, out, out])
    out2=cv2.vconcat([out1,out1,out1])
    out2 = out2[250:650,250:650]
    return out2


def zoom_out(img):
    y_s = 600
    x_s = 600
    out = img.resize((int(x_s), int(y_s)), Image.ANTIALIAS)  # resize image with high-quality
    out = np.asarray(out)
    out1 = cv2.hconcat([out, out, out])
    out2 = cv2.vconcat([out1, out1, out1])
    out2 = out2[100:500, 100:500]
    return out2

def color(image_filename):
    img = cv2.imread(image_filename)
    h, s, v = cv2.split(img)
    v1 = np.clip(cv2.add(1 * v, 30), 0, 255)
    v2 = np.clip(cv2.add(2 * v, 20), 0, 255)

    img1 = np.uint8(cv2.merge((h, s, v1)))
    img_bri = cv2.cvtColor(img1, cv2.COLOR_HSV2BGR)
    
    return img_bri


def sharpen(image_filename):
    sharp_kernel = np.array([
        [-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1]
    ])
    img = cv2.imread(image_filename)

    img = cv2.filter2D(img, -1, sharp_kernel)
    return img

def data_processing(image_filename, label): #label = 1 augmentated images; label = 0 augmentated masks
    aug_imgs = []
    aug_masks = []
    tmp_augimgs = []
    tmp_augmasks = []
    img = Image.open(image_filename)
    tmp_augimgs.append(img)
    tmp_augmasks.append(img)
    # flip
    img_flip1 = H_flip(img)
    tmp_augimgs.append(img_flip1)
    tmp_augmasks.append(img_flip1)
    img_flip2 = V_flip(img)
    tmp_augimgs.append(img_flip2)
    tmp_augmasks.append(img_flip2)
    #random rotate
    img_rotate = RandomRotate(image_filename)
    tmp_augimgs.append(img_rotate)
    tmp_augmasks.append(img_rotate)
    #sharp
    img_sh = sharpen(image_filename)
    tmp_augimgs.append(img_sh)
    #color
    img_bri = color(image_filename)
    tmp_augimgs.append(img_bri)
    # add two masks in folder
    for i in range(2):
        tmp_augmasks.append(img)
    #scale
    img_zoom = scale(img)
    tmp_augimgs.append(img_zoom)
    tmp_augmasks.append(img_zoom)
    img_zoomin = zoom_in(img)
    tmp_augimgs.append(img_zoomin)
    tmp_augmasks.append(img_zoomin)
    img_zoomout = zoom_out(img)
    tmp_augimgs.append(img_zoomout)
    tmp_augmasks.append(img_zoomout)
    n = len(tmp_augimgs)

    for i in range(n):
        aug_imgs.append(np.asarray(tmp_augimgs[i]) / 255)
        aug_masks.append(np.asarray(tmp_augmasks[i]) / 255)
    if label == 1:
        return aug_imgs
    if label == 0:
        return aug_masks
    
    









