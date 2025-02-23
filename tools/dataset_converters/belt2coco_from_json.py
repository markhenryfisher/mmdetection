# -*- coding: utf-8 -*-
"""
Shpw belt data annotation (UEA json annotation format).
e.g. Usage: 
    python belt_data_browser.py "./data/ruth/datasets/belt_data_natural" "MRV SCOTIA" 

Created on Thu Feb  1 10:44:30 2024

@author: mhf@uea.ac.uk
@filename: belt2coco_from_json.py
last update: 06.02.24
"""
import argparse
import os
import numpy as np
import cv2
import random
from PIL import Image
from mmengine.fileio import load, dump
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import pdb

# minimum area of a region (smaller areas raise an error)
MIN_AREA = 20

############## folder renaming utils ##########
# Constants for file / folder renaming
IMAGES_FOLDER_NAME = 'label_frames'
LABELS_FOLDER_NAME = 'seg_labels_json'
IMAGE_SUFFIX = '.png'
LABEL_SUFFIX = '__labels.json'

# label types
PRIMITIVE_LABELS = ['polygon']
COMPOUND_LABELS = ['group']

def label_to_image(label):
    return label.replace(LABEL_SUFFIX, IMAGE_SUFFIX).replace(LABELS_FOLDER_NAME, IMAGES_FOLDER_NAME)

def image_to_label(image):
    return image.replace(IMAGE_SUFFIX, LABEL_SUFFIX).replace(IMAGES_FOLDER_NAME, LABELS_FOLDER_NAME)

############### arg parser ##################
def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert json annotation to coco format')
    parser.add_argument('dataset', nargs='?', help='belt data file path')
    parser.add_argument('belt', nargs='?', help='belt name, e.g. MRV SCOTIA')
    parser.add_argument('out_folder', nargs='?', help='output folder')

    args = parser.parse_args()

    return args

############### image plotting ##############
    
def display(img, contours=None, show=True):
    """
    render / display an image (with optional contour plot)

    Parameters
    ----------
    img : TYPE
        DESCRIPTION.
    contours : TYPE, optional
        DESCRIPTION. The default is None.
    show : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    out_img : np array (RGB image)
        DESCRIPTION.

    """
    if img.dtype == np.bool_:
        # convert binay mask to image
        gray = img.astype(np.float32)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)       
        
    h, w = img.shape[:2]
    
    # Display the image and plot all contours found
    fig, ax = plt.subplots()

    # Display image so it nicely fills the figure
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)    
    # Get the size of the data
    dim = np.array(img.shape[:2][::-1]) + np.array([0, 0.5])
    # Set the figure sizes in inches
    fig.set_size_inches(dim / fig.dpi)
   
    ax.imshow(img)
    if contours is not None:
        # ax.imshow(img)
        for c in contours:
            px = c[:, 0]; py = c[:, 1]
            
            x_min = np.min(px)
            x_max = np.max(px)
            y_min = np.min(py)
            y_max = np.max(py)
                
            # Create a Rectangle patch
            rect = patches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, linewidth=2, edgecolor='r', facecolor='none')

            # Add the patch to the Axes
            ax.add_patch(rect)
                
            ax.plot(px, py, 'w', linewidth=2)
    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])

    
    s, (width, height) = fig.canvas.print_to_buffer() 
    
    # Convert to a NumPy array.
    out_img = np.frombuffer(s, np.uint8).reshape((height, width, 4))    
    out_img = cv2.cvtColor(out_img, cv2.COLOR_RGBA2RGB)
    out_img = cv2.resize(out_img, (w, h))

    if show:    
        plt.show()
    
    plt.close(fig)
        
    return out_img

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
    
    train_list = [img_list[x] for x in train_idxs]
    val_list = [img_list[x] for x in test_idxs]
    
    return train_list, val_list


def fix_polygon_vertices(regions, img_size):
    
    def clip_polygon(region, img_size):
        w, h = img_size 
        region[:,0] = np.clip(region[:,0], a_min=0, a_max=w-1)
        region[:,1] = np.clip(region[:,1], a_min=0, a_max=h-1)
        
        return region
    
    def close_polygon(region):    
        region = np.rint(region)
        # close the contour
        closed = region[0,:] == region[-1,:]
        if not closed.all():
            region = np.vstack([region, region[0,:]])
            
        return region
                
    for i, region in enumerate(regions):
        region = close_polygon(region)
        region = clip_polygon(region, img_size)
        regions[i] = region
        
    return regions




def unpack_polygon(label_json):
    
    def verify_area(regions):
        filtered_regions = []
        for c in regions:
            px = c[:, 0]; py = c[:, 1]
            
            x_min = np.min(px)
            x_max = np.max(px)
            y_min = np.min(py)
            y_max = np.max(py)
            
            area = (x_max - x_min) * (y_max - y_min)
            
            if area < MIN_AREA:
                print('Unpacking Error: Region area too small')
            else:
                filtered_regions.append(c)

        return filtered_regions
        
    def verify_regions(regions_json):
        for region in regions_json:
            if len(region) > 3:
                pass
            else:
                raise RuntimeError('Unpacking Error: Insufficient Vertices')                              
        return
       
    if 'vertices' in label_json:
        regions_json = [label_json['vertices']]
    else:
        regions_json = label_json['regions']

    verify_regions(regions_json)        
    regions = [np.array([[v['x'], v['y']] for v in region_json]) for region_json in regions_json]
    regions = verify_area(regions)

    return regions
      

def convert_label(mask, label, debug=False):
    h, w = mask.shape[:2]
        
    # simple polygon
    if label['label_type'] in PRIMITIVE_LABELS:
        print('{} {}'.format(label['label_type'], label['object_id']))
        regions = unpack_polygon(label)
        regions = fix_polygon_vertices(regions, (w,h))
        if debug:
            mask = display(mask, regions, show=True)
    # group of polygon 'components'
    elif label['label_type'] in COMPOUND_LABELS:
        print(label['label_type'])
        print(label['object_id'])
        for component in label['component_models']:
            print(component['label_type'])
            print(component['object_id'])
            regions = convert_label(mask, component)
    else:
        raise RuntimeError('Unknown label type')
        
    return regions
        
    
def convert2coco(img_list, img_folder, out_file):
    
    annotations = []
    images = []
    obj_count = 0  
    
    for idx in range(len(img_list)):
    # for idx in range(0,7):
        img_filename = img_list[idx]
        print('idx: {}, File name: {}'.format(idx, img_filename))
        img_path = os.path.join(img_folder, img_filename)
        lbl_path = image_to_label(img_path) 
      
        img = Image.open(img_path).convert("RGB")
        width, height = img.size 

        
        labels = load(lbl_path)
        for i, label in enumerate(labels):
        #for i in range(6,7):
            label = labels[i]
            has_poly = False
            regions = convert_label(np.array(img), label)
                
                
            for c in regions:
                has_poly = True
                px = c[:, 0]; py = c[:, 1]
                
                poly = [(x, y) for x, y in zip(px, py)]
                poly = [p for x in poly for p in x]

                
                x_min = np.min(px)
                x_max = np.max(px)
                y_min = np.min(py)
                y_max = np.max(py)

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
    args = parse_args()    
    # if args.dataset is None:
    #     args.dataset = './data/ruth/datasets/belt_data_natural'
    # if args.belt is None: 
    #     args.belt = 'MRV SCOTIA'
    # if args.out_folder is None:
    #     args.out_folder = './data/belt_data_natural'

    if args.dataset is None:
        args.dataset = './data/mhf/datasets/belt_data_synthetic/Experiment160'
    if args.belt is None: 
        args.belt = 'mrv_scotia'
    if args.out_folder is None:
        args.out_folder = './data/belt_data_synthetic/Experiment160'



    

    if os.path.exists(os.path.join(args.dataset, 'splits')):
        belt_prefix = os.path.join(args.out_folder, args.belt)

        train_image_folder = os.path.join(args.dataset, 'label_frames', 'train_'+args.belt)
        train_annotation_folder = os.path.join(args.dataset, 'seg_labels_json', 'train_'+args.belt)

        val_image_folder = os.path.join(args.dataset, 'label_frames', 'test_'+args.belt)
        val_annotation_folder = os.path.join(args.dataset, 'seg_labels_json', 'test_'+args.belt)

        if os.path.exists(train_image_folder):
            train_list = list(sorted(os.listdir(train_image_folder)))
            # make a dirs for coco train
            train_prefix = os.path.join(belt_prefix, 'train')
            if not os.path.exists(train_prefix):
                os.makedirs(train_prefix)

            convert2coco(train_list, 
                          train_image_folder,
                          out_file = os.path.join(train_prefix, 'annotation_coco.json'))

        
        if os.path.exists(val_image_folder):
            val_list = list(sorted(os.listdir(val_image_folder)))

            # make a dirs for coco test
            val_prefix = os.path.join(belt_prefix, 'val')
            if not os.path.exists(val_prefix):
                os.makedirs(val_prefix)   
       
            convert2coco(val_list, 
                          val_image_folder, 
                          out_file = os.path.join(val_prefix, 'annotation_coco.json'))
     

    else:
        image_folder = os.path.join(args.dataset, 'label_frames', args.belt)
        annotation_folder = os.path.join(args.dataset, 'seg_labels_json', args.belt)
        img_list = list(sorted(os.listdir(image_folder)))
        train_list, val_list = generate_splits(img_list)
    
        # make a dirs for coco train/test
        belt_prefix = os.path.join(args.out_folder, args.belt)
        train_prefix = os.path.join(belt_prefix, 'train')
        if not os.path.exists(train_prefix):
            os.makedirs(train_prefix)
        val_prefix = os.path.join(belt_prefix, 'val')
        if not os.path.exists(val_prefix):
            os.makedirs(val_prefix)   
            
             
        convert2coco(train_list, 
                      image_folder,
                      out_file = os.path.join(train_prefix, 'annotation_coco.json'))
    
        convert2coco(val_list, 
                      image_folder, 
                      out_file = os.path.join(val_prefix, 'annotation_coco.json'))

     
