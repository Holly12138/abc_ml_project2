import os
import matplotlib.image as mpimg
from sklearn.metrics import f1_score
from PIL import Image
import numpy as np


def patch_to_label(patch, args):
     # 25% percentage of pixels > 1 required to assign a foreground label to a patch
    df = np.mean(patch)
    if df > 0.25:
        return 1
    else:
        return 0

def classifier(im, args, w=16, h=16):
    list_labels = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            label = patch_to_label(im_patch, args)
            list_labels.append(label)
    return list_labels

# compute F1
def F1(pred, label, args):
    """compute the f1 score"""
    patch_pred = [classifier(pred[i].cpu().detach().numpy(), args) for i in range(len(pred))]
    patch_label = [classifier(label[i].cpu().detach().numpy(), args) for i in range(len(pred))]
    f1 = f1_score(np.array(patch_label).reshape(-1), np.array(patch_pred).reshape(-1))
    return f1