# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 10:44:30 2024

@author: mhf
"""
import os
import numpy as np
import cv2
from PIL import Image
from image_labelling_tool import labelling_tool
from mmengine.fileio import dump, load
import matplotlib.pyplot as plt
import matplotlib.patches as patches

BELTS = ['MRV SCOTIA']

# All fish classes found within my dataset, used for segmentation
FISH_CLASSES = ['fish_mackerel', 'fish_redgurnard', 'fish_catfish', 'fish_gurnard', 'fish_haddock', 'fish_ling',
                'fish_lemonsole', 'fish_monk', 'fish_dogfish', 'fish_commondab', 'fish_squid', 'fish_megrim',
                'fish_doversole', 'fish_herring', 'fish_unknown', 'fish_small', 'fish_horsemackerel', 'fish_argentines',
                'fish_skate_ray', 'fish_longroughdab', 'fish_plaice', 'fish_greygurnard', 'fish_flat_generic',
                'fish_partial', 'fish_whiting', 'fish_saithe', 'fish_norwaypout', 'fish_misc', 'fish_bib',
                'fish_boar_fish', 'fish', 'whole_fish', 'fish_seabass', 'fish_commondragonet', 'fish_brill',
                'fish_cod', 'fish_hake', 'fish_john_dory', 'fish_multiple']


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
    else:
        fig.clear()
        
    return out_img

def clip_to_image(regions, img_size):
    w, h = img_size 
    for i, region in enumerate(regions):
        region[:,0] = np.clip(region[:,0], a_min=0, a_max=w-1)
        region[:,1] = np.clip(region[:,1], a_min=0, a_max=h-1)
        regions[i] = region
    
    return regions
    
# def unpack_polygon(label, img_size):
#     regions_json = [label['vertices']]
#     regions = [np.array([[v['x'], v['y']] for v in region_json]) for region_json in regions_json]
#     for i, region in enumerate(regions):
#         regions[i] = clip_to_image(region, img_size)
    
#     return regions

def unpack_polygon(label_json):
    if 'vertices' in label_json:
        regions_json = [label_json['vertices']]
    else:
        regions_json = label_json['regions']
        
    regions = [np.array([[v['x'], v['y']] for v in region_json]) for region_json in regions_json]

    return regions
      

def display_label(mask, label):
    h, w = mask.shape[:2]
        
    # simple polygon
    if label['label_type'] in PRIMITIVE_LABELS:
        print('{} {}'.format(label['label_type'], label['object_id']))
        regions = unpack_polygon(label)
        regions = clip_to_image(regions, (w,h))
        mask = display(mask, regions, show=False)
    # group of polygon 'components'
    elif label['label_type'] in COMPOUND_LABELS:
        print(label['label_type'])
        print(label['object_id'])
        for component in label['component_models']:
            print(component['label_type'])
            print(component['object_id'])
            mask = display_label(mask, component)
    else:
        raise RuntimeError('Unknown label type')
        
    return mask
        

def show_annotation(img_folder, annotation_folder, img_filename):
    img_path = os.path.join(img_folder, img_filename)
    lbl_path = image_to_label(img_path) 
    
    limg = labelling_tool.PersistentLabelledImage(img_path, lbl_path)
    mask, cls_map = limg.render_label_instances(FISH_CLASSES, multichannel_mask=False)

    # labels = labelling_tool.ImageLabels.from_file(lbl_path)

    # load image
    img = Image.open(img_path).convert("RGB")
    width, height = img.size 
    
    
    labels = load(lbl_path)
    for label in labels:
        img = display_label(np.array(img), label)
        
    display(img)
    plt.close()
    
    

if __name__ == '__main__':
    image_prefix = './data/ruth/datasets/belt_data_natural/label_frames'
    annotation_prefix = './data/ruth/datasets/belt_data_natural/seg_labels_json'
    out_prefix = './data/belt_data_natural'
    
    for belt in BELTS:
        img_folder = os.path.join(image_prefix, belt)
        annotation_folder = os.path.join(annotation_prefix, belt)
        img_list = list(sorted(os.listdir(img_folder)))
        
        for i in range(len(img_list)):
            img_filename = img_list[i]
            show_annotation(img_folder, annotation_folder, img_filename)
