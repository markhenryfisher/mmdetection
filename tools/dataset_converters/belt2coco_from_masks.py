# -*- coding: utf-8 -*-
"""
belt_data_natural dataset converter
Created on Mon Jan 22 13:12:42 2024

bugs: Only handles annotations involving single regions. See line 86.

Note: Launch in working directory 'mmdetection'
@author: mhf@uea.ac.uk
@filename: belt2coco_from_masks.py
@last_updated: 06.02.24
"""
import os
import cv2
import random
import numpy as np
import scipy as sp
import warnings
from PIL import Image
from skimage import measure, morphology
from image_labelling_tool import labelling_tool
from mmengine.fileio import dump
from mmengine.utils import track_iter_progress

import matplotlib.pyplot as plt

# BELTS = ['BOY JOHN INS110', 'Carhelmar', 'CRYSTAL SEA SS118',
#          'GLENUGIE PD347', 'LAPWING PD972', 'MRV SCOTIA', 'SUMMER DAWN PD97']

BELTS = ['MRV SCOTIA']


# All fish classes found within my dataset, used for segmentation
FISH_CLASSES = ['fish_mackerel', 'fish_redgurnard', 'fish_catfish', 'fish_gurnard', 'fish_haddock', 'fish_ling',
                'fish_lemonsole', 'fish_monk', 'fish_dogfish', 'fish_commondab', 'fish_squid', 'fish_megrim',
                'fish_doversole', 'fish_herring', 'fish_unknown', 'fish_small', 'fish_horsemackerel', 'fish_argentines',
                'fish_skate_ray', 'fish_longroughdab', 'fish_plaice', 'fish_greygurnard', 'fish_flat_generic',
                'fish_partial', 'fish_whiting', 'fish_saithe', 'fish_norwaypout', 'fish_misc', 'fish_bib',
                'fish_boar_fish', 'fish', 'whole_fish', 'fish_seabass', 'fish_commondragonet', 'fish_brill',
                'fish_cod', 'fish_hake', 'fish_john_dory', 'fish_multiple']
# All fish classes which could be used in the synthetic images (whole, manually annotated examples in dataset). Used
# for classification
CLS_FISH_CLASSES = ['fish_mackerel', 'fish_redgurnard', 'fish_catfish', 'fish_haddock', 'fish_ling',
                    'fish_lemonsole', 'fish_monk', 'fish_dogfish', 'fish_commondab', 'fish_megrim',
                    'fish_doversole', 'fish_herring', 'fish_horsemackerel', 'fish_argentines',
                    'fish_skate_ray', 'fish_longroughdab', 'fish_plaice', 'fish_greygurnard',
                    'fish_whiting', 'fish_saithe', 'fish_norwaypout', 'fish_bib', 'fish_boar_fish', 'fish_brill',
                    'fish_seabass', 'fish_cod', 'fish_hake', 'fish_john_dory', 'fish_commondragonet']



def generate_splits(img_list):
    # randomize dataset and split into train and test 
    n_train = 9 * len(img_list) // 10 
    n_test = len(img_list) - n_train  
    idxs = list(range(len(img_list)))
    # get the same random behaviour every time
    random.seed(1)
    random.shuffle(idxs)
    train_idxs = idxs[:n_train]
    test_idxs = idxs[-n_test:]  
    
    return train_idxs, test_idxs

def remove_border(mask):
    """
    masks that touch the border are problematic 
    """
    # force border to 0 (to make contour complete)
    height, width = mask.shape    
    mask[0, :] = 0; mask[height-1, :] = 0 
    mask[:, 0] = 0; mask[:, width-1] = 0
    
    return mask

def get_contour(mask, debug=True):
    
    contours = measure.find_contours(mask, fully_connected='high')
      
    # No contours    
    if len(contours) == 0:
        return None
    
    # simplify and close contours
    for idx, c in enumerate(contours):
        # round to nearest integer
        c = np.rint(c)
        # close the contour
        closed = c[0] == c[-1]
        if not closed.all():
            c = np.append(c, c[0])       
        # simplify the contour, tollerance = 1.0 pixel
        contours[idx] = measure.approximate_polygon(c, 2.0)
        
    if debug:     
        display(mask, contours)
        
    return contours
     

def poly2coco(mask):
    """
    get polygon contours, and convert to coco format

    """ 
    contours = get_contour(mask)
    if contours is None:
        return None
    elif len(contours) > 1:
        print('Two contours')
    for idx, c in enumerate(contours):
        px = c[:, 1]; py = c[:, 0]        
        # poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
        poly = [(x, y) for x, y in zip(px, py)]
        poly = [p for x in poly for p in x]
        contours[idx] = poly
    
    return contours

def filter_masks(mask, idx, threshold=30, debug=False):
    """
    Clean up output of image labeller:
    1. Split the color-encoded mask into a set
    of binary masks.
    2. Fill holes in binary masks.
    3. Remove small objects (area <= threshold).
    
    """ 
    # remove the border
    mask = remove_border(mask)
    # instances are encoded as different colors
    obj_ids = np.unique(mask)
    # first id is the background, so remove it
    obj_ids = obj_ids[1:]

    # split the color-encoded mask into a set
    # of binary masks
    masks = mask == obj_ids[:, None, None] 
    
    ok_masks = []
    for mask in masks:
        # fill holes
        mask = sp.ndimage.binary_fill_holes(mask)
        
        # re-label mask
        sub_mask = measure.label(mask, background=0)

        props = measure.regionprops(sub_mask.astype(np.uint8))
        
        temp_mask = np.zeros_like(mask)
        for p in props:
            area = p.area
            if area > threshold:
                lbl = p.label
                temp_mask[sub_mask==lbl] = True

            else:
                warnings.warn("Index={}: Removing small object, area = {}".format(idx, area))

        # add temp_mask to ok_masks 
        if np.any(temp_mask):
            if debug:
                display(temp_mask)
            ok_masks.append(temp_mask)            
            
        
    out_obj_ids = np.array((list(range(1, len(ok_masks)+1))), dtype=np.int32) 
    out_masks = np.array((ok_masks))
    
    return out_masks, out_obj_ids
        

def convert2coco(idxs, image_prefix, annotation_prefix, out_file, debug=True):
    img_list = list(sorted(os.listdir(image_prefix)))

    annotations = []
    images = []
    obj_count = 0       
    #for n, idx in enumerate(track_iter_progress(idxs)):

    for n in range(10,11):
        idx = idxs[n]        
        print('n={}, filename: {}'.format(n, img_list[idx]))
        # load images
        img_filename = img_list[idx]
        img_path = os.path.join(image_prefix, img_filename)
        img = Image.open(img_path).convert("RGB")
        width, height = img.size 
        
        # render annotation
        basename, __ = os.path.splitext(img_filename)
        json_filename = basename + '__labels.json'
        label_path = os.path.join(annotation_prefix, json_filename)
        
        limg = labelling_tool.PersistentLabelledImage(img_path, label_path)
        mask, cls_map = limg.render_label_instances(FISH_CLASSES, multichannel_mask=False)

        if debug:
            # convert binay mask to image
            grey_img = mask.astype(np.float64)
            
            # Display the image and plot all contours found
            fig, (ax1, ax2) = plt.subplots(1,2)
            ax1.imshow(img)
            ax1.axis('image')
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax2.imshow(grey_img, cmap=plt.cm.gray)
            ax2.axis('image')
            ax2.set_xticks([])
            ax2.set_yticks([])
            plt.show()

        # perform an error check on mask and (optionally) fix any problems
        # and return a set of binary masks
        masks, obj_ids = filter_masks(mask, idx)
        
        # set flag in case contours are unusable
        has_poly = False
        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        for i in range(num_objs):
            mask = masks[i]
            coco_contours = poly2coco(mask)
            # if mask has polygons
            if coco_contours is not None: 
                has_poly = True
                
               
                # for i, contour in enumerate(coco_contours):
                #     sub_mask = sub_masks[i]
                   
                pos = np.nonzero(mask)
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
                    segmentation=coco_contours,
                    iscrowd=0)
                
                annotations.append(data_anno)
                obj_count += 1
            
        # image contains at least one contour
        if has_poly:
            # add image to list
            basename, __ = os.path.splitext(img_filename)
            jpg_filename = basename + '.jpg'
            images.append(dict(id=idx, file_name=jpg_filename, height=height, width=width))
            # save jpg image 
            fullfile = os.path.join(os.path.dirname(out_file), jpg_filename)
            if not os.path.isfile(fullfile):
                img.save(fullfile)
                
            
            
    coco_format_json = dict(
    images=images,
    annotations=annotations,
    categories=[{
        'id': 0,
        'name': 'fish_unknown'
    }])
    
    dump(coco_format_json, out_file)
    
############### image plotting ##############
    
def display(img, contours=None):
    if img.dtype == np.bool_:
        # convert binay mask to image
        gray = img.astype(np.float32)
        img = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
    
    # Display the image and plot all contours found
    fig, ax = plt.subplots()
    ax.imshow(img)
    if contours is not None:
        # ax.imshow(img)
        for c in contours:
            px = c[:, 1]; py = c[:, 0]
            ax.plot(px, py, 'b', linewidth=2)
    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
    

if __name__ == '__main__':
    image_prefix = './data/ruth/datasets/belt_data_natural/label_frames'
    annotation_prefix = './data/ruth/datasets/belt_data_natural/seg_labels_json'
    out_prefix = './data/belt_data_natural'
    
    for belt in BELTS:
        input_image_prefix = os.path.join(image_prefix, belt)
        input_annotation_prefix = os.path.join(annotation_prefix, belt)
        img_list = list(sorted(os.listdir(input_image_prefix)))
        train_idxs, val_idxs = generate_splits(img_list)
        
        # make a dirs for coco train/test
        belt_prefix = os.path.join(out_prefix, belt)
        train_prefix = os.path.join(belt_prefix, 'train')
        if not os.path.exists(train_prefix):
            os.makedirs(train_prefix)
        val_prefix = os.path.join(belt_prefix, 'val')
        if not os.path.exists(val_prefix):
            os.makedirs(val_prefix)
            
        convert2coco(train_idxs, 
                      input_image_prefix, 
                      input_annotation_prefix,
                      out_file = os.path.join(train_prefix, 'annotation_coco.json'))

        convert2coco(val_idxs, 
                     input_image_prefix, 
                     input_annotation_prefix,
                     out_file = os.path.join(val_prefix, 'annotation_coco.json'))

        