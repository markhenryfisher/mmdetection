# -*- coding: utf-8 -*-
"""
belt_data_natural dataset converter
Created on Mon Jan 22 13:12:42 2024

bugs: Only handles annotations involving single regions. See line 86.

Note: Launch in working directory 'mmdetection'
@author: mhf
@filename: belt_data_natural2coco.py
@last_updated: 24.01.24
"""
import os
import random
import numpy as np
from PIL import Image
from skimage import measure
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
    # force border to 0 (to make contour complete)
    height, width = mask.shape    
    mask[0, :width] = 0; mask[height-1, :width] = 0 
    mask[:height, 0] = 0; mask[:height, width-1] = 0
    
    return mask

def contour(mask, debug=False):
    """
    https://scikit-image.org/docs/stable/auto_examples/edges/plot_contours.html

    """    
    # convert binay mask to image
    grey_img = mask.astype(np.float64)
    
    # Find contours at a constant value of 0.8 (arbitrary since mask is binary)
    # Note: A mask may have several contours (mask may include a hole) 
    contours = measure.find_contours(grey_img)
    
    # simplify the contour
    
    
    # don't yet know how to handle multiple contours
    if len(contours) == 0 or len(contours) > 1:
        return None
    
    coords = contours[0]
    # simplify the contour, tollerance = 1.0 pixel
    coords = measure.approximate_polygon(coords, 1.0)
    px = coords[:, 1]; py = coords[:, 0]
                        
    if debug:     
        # Display the image and plot all contours found
        fig, ax = plt.subplots()
        ax.imshow(grey_img, cmap=plt.cm.gray)
        ax.plot(px, py, linewidth=2)
        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()
        
    return (px, py)

    

def convert2coco(idxs, image_prefix, annotation_prefix, out_file):
    img_list = list(sorted(os.listdir(image_prefix)))

    annotations = []
    images = []
    obj_count = 0       
    for n, idx in enumerate(track_iter_progress(idxs)):
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
       
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None] 
       
        # set flag in case contours are unusable
        has_poly = False
        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        for i in range(num_objs):
            mask = remove_border(masks[i])
            
            coords = contour(mask)
            # check image contains polygons
            if coords is not None: 
                has_poly = True
                
                px, py = coords          
                # poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                poly = [(x, y) for x, y in zip(px, py)]
                poly = [p for x in poly for p in x]
                   
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
                    segmentation=[poly],
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

        