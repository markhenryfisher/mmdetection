# -*- coding: utf-8 -*-
"""
PennFudanPed dataset converter.

Created on Thu Jan 18 15:24:46 2024

Note: Launch in working directory 'mmdetection'
@author: mhf
@filename: pennfudenped2coco.py
@last_updated: 19.01.24
"""
from mmengine.fileio import dump
from mmengine.utils import track_iter_progress

import os
import numpy as np
from PIL import Image
from skimage import measure
import random

import matplotlib.pyplot as plt


def contour(mask, debug=False):
    """
    https://scikit-image.org/docs/stable/auto_examples/edges/plot_contours.html

    """
    # convert binay mask to image
    grey_img = mask.astype(np.float64)
    
    # Find contours at a constant value of 0.8 (arbitrary since mask is binary)
    # Note: A mask may have several contours (mask may include a hole) 
    contours = measure.find_contours(grey_img, 0.8)
    
    # how to handle a mask with a hole?
    # only need ONE (longest) contour
    if len(contours) > 1:
        c = max(contours, key=len)
    else:
        c = contours[0]
    px = c[:, 1]; py = c[:, 0]
                        
    if debug:     
        # Display the image and plot all contours found
        fig, ax = plt.subplots()
        ax.imshow(grey_img, cmap=plt.cm.gray)
        ax.plot(px, py, linewidth=2)
        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()
        
    return px, py

def convert_pennfudenped2coco(idxs, image_prefix, mask_prefix, out_file):
    """

    """ 
    img_list = list(sorted(os.listdir(image_prefix)))
    mask_list = list(sorted(os.listdir(mask_prefix)))
    
    annotations = []
    images = []
    obj_count = 0   
    # for idx in track_iter_progress(range(len(img_list))):
    for idx in track_iter_progress(idxs):

        # load images and masks
        img_path = os.path.join(image_prefix, img_list[idx])
        mask_path = os.path.join(mask_prefix, mask_list[idx])
        
        filename = img_list[idx]
        img = Image.open(img_path).convert("RGB")
        width, height = img.size        
        
        # save the image as jpg
        basename, ext = os.path.splitext(filename)
        jpg_filename = basename +'.jpg'
        fullfile = os.path.join(os.path.dirname(out_file), jpg_filename)
        img.save(fullfile)
        
        images.append(
            dict(id=idx, file_name=jpg_filename, height=height, width=width))


        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        for i in range(num_objs):
            px, py = contour(masks[i])
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            pos = np.nonzero(masks[i])
            x_min = np.min(pos[1])
            x_max = np.max(pos[1])
            y_min = np.min(pos[0])
            y_max = np.max(pos[0])
            
            data_anno = dict(
                image_id=idx,
                id=obj_count,
                category_id=0,
                bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                area=(x_max - x_min) * (y_max - y_min),
                segmentation=[poly],
                iscrowd=0)
            
            annotations.append(data_anno)
            obj_count += 1
            
    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=[{
            'id': 0,
            'name': 'PASpersonWalking'
        }])
    dump(coco_format_json, out_file)

    
if __name__ == '__main__':
    mask_prefix = './data/PennFudanPed/PedMasks'
    image_prefix = './data/PennFudanPed/PNGImages' 
    
    img_list = list(sorted(os.listdir(image_prefix)))
    # mask_list = list(sorted(os.listdir(mask_prefix)))
    
    # randomize dataset and split into train and test 
    n_train = 9 * len(img_list) // 10 
    n_test = len(img_list) - n_train  
    idxs = list(range(len(img_list)))
    # get the same random behaviour every time
    random.seed(1)
    random.shuffle(idxs)
    train_idxs = idxs[:n_train]
    test_idxs = idxs[-n_test:]
    
    # make a dirs for coco train/test
    if not os.path.exists('./data/PennFudanPed/train'):
        os.makedirs('./data/PennFudanPed/train')
    if not os.path.exists('./data/PennFudanPed/val'):
        os.makedirs('./data/PennFudanPed/val')        
    
    
    convert_pennfudenped2coco(train_idxs, 
                              image_prefix, 
                              mask_prefix, 
                              out_file = './data/PennFudanPed/train/annotation_coco.json')
    
    convert_pennfudenped2coco(test_idxs, 
                              image_prefix, 
                              mask_prefix, 
                              out_file = './data/PennFudanPed/val/annotation_coco.json')


    

