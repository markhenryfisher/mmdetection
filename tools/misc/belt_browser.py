# -*- coding: utf-8 -*-
"""
Show and fix belt data annotation (UEA json annotation format).
e.g. Usage: 
    python belt_data_browser.py "./data/ruth/datasets/belt_data_natural" "MRV SCOTIA"
or to fix labels and overwrite (warning!!! backup original first)   
    python belt_data_browser.py "./data/ruth/datasets/belt_data_natural" "MRV SCOTIA" -o 
        
Created on Thu Feb  1 10:44:30 2024

@author: mhf@uea.ac.uk
@filename: belt_browser.py
last update: 07.02.24
"""
import argparse
import os
import numpy as np
import cv2
from PIL import Image
from mmengine.fileio import load, dump
import matplotlib.pyplot as plt
import matplotlib.patches as patches

############## region verify area threshold ###
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
        description='Browse and confirm catchmonitor annotations')
    parser.add_argument('dataset', nargs='?', help='belt data file path')
    parser.add_argument('belt', nargs='?', help='belt name, e.g. MRV SCOTIA')
    parser.add_argument('-o', '--overwrite',
                    action='store_true')  # on/off flag

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
        for c in regions:
            px = c[:, 0]; py = c[:, 1]
            
            x_min = np.min(px)
            x_max = np.max(px)
            y_min = np.min(py)
            y_max = np.max(py)
            
            area = (x_max - x_min) * (y_max - y_min)
            
            if area < MIN_AREA:
                print('Unpacking Error: Region area too small')
                low_area = True
            else:
                low_area = False
        return low_area
        
    def verify_regions(regions_json):
        for region in regions_json:
            if len(region) > 3:
                low_vertex = False
            else:
                print('Unpacking Error: Insufficient Vertices')
                low_vertex = True                              
        return low_vertex
        
    if 'vertices' in label_json:
        regions_json = [label_json['vertices']]
    else:
        regions_json = label_json['regions']

    vertex_error = verify_regions(regions_json)        
    regions = [np.array([[v['x'], v['y']] for v in region_json]) for region_json in regions_json]
    area_error = verify_area(regions)
    
    if vertex_error or area_error:
        regions = []
    
    return regions      

def display_label(mask, label):
    h, w = mask.shape[:2]
        
    # simple polygon
    if label['label_type'] in PRIMITIVE_LABELS:
        print('{} {}'.format(label['label_type'], label['object_id']))
        regions = unpack_polygon(label)
        if regions:
            regions = fix_polygon_vertices(regions, (w,h))
            mask = display(mask, regions, show=False)
        else:
            label = []
    # group of polygon 'components'
    elif label['label_type'] in COMPOUND_LABELS:
        print(label['label_type'])
        print(label['object_id'])
        out_components = []
        for component in label['component_models']:
            print(component['label_type'])
            print(component['object_id'])
            mask, regions = display_label(mask, component)
            if not regions:
                print('Error in {}: {}'.format(label['label_type'], label['object_id']))
            else:
                out_components.append(component)
        label['component_models'] = out_components
    else:
        raise RuntimeError('Unknown label type')        
        
    return mask, label
        

def show_annotation(img_folder, annotation_folder, img_filename, overwrite):
    out_labels_json = []
    
    img_path = os.path.join(img_folder, img_filename)
    lbl_path = image_to_label(img_path) 
    
    # load image
    img = Image.open(img_path).convert("RGB")
    width, height = img.size 
        
    labels = load(lbl_path)
    for idx, label in enumerate(labels):
        img, label = display_label(np.array(img), label)
        if label:
            out_labels_json.append(label)
        else:
            pass
                    
    display(img)
    plt.close()

    if overwrite:
        print('Overwriting: {}'.format(img_filename))
        dump(out_labels_json, lbl_path) 
 
    
if __name__ == '__main__':
    args = parse_args()    
    if args.dataset is None:
        args.dataset = './data/ruth/datasets/belt_data_natural/'
    if args.belt is None: 
        args.belt = 'MRV SCOTIA'
    overwrite = args.overwrite

    
    image_folder = os.path.join(args.dataset, 'label_frames', args.belt)
    annotation_folder = os.path.join(args.dataset, 'seg_labels_json', args.belt)
    
    img_list = list(sorted(os.listdir(image_folder)))
    
    print('Showing {} annotations'.format(len(img_list)))
    input('Press ENTER to continue: ')
    

    reject_images = []
    for i in range(len(img_list)):
        img_filename = img_list[i]              
        show_annotation(image_folder, annotation_folder, img_filename, overwrite)
        user_input = input('Accept {} ? [default: y)es]:  '.format(img_filename))
        if len(user_input) == 0 or user_input in ['y', 'yes']: 
            user_input = 'y'
        else:
            reject_images.append(img_filename)
            
        if i > 0 and i % 10 == 0:
            print('\n\nCurrent rejects:')
            for reject in reject_images:
                print('index: {}, File: {}'.format(i, reject))
            user_input = input('Continue ? [default: y)es]: ' )
            if len(user_input) == 0 or user_input in ['y', 'yes']: 
                pass
            else:
                break
                
            
            
    print('\n\nFinal rejects:')
    for reject in reject_images:
        print(reject)
    print('\n')   
