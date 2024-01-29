# -*- coding: utf-8 -*-
"""
Attempt to write a script to separate human and machine annotation.
Tests concluded this was impossible.

Created on Mon Jan 29 10:24:43 2024

@author: mhf
"""
import os
from mmengine.fileio import dump, load

BELTS = ['MRV SCOTIA']

def write_file(file_handle, txt):
    file_handle.write(txt)
    
    return


def get_stats(info):
    nx = 1000; mx = 0; ny = 1000; my = 0
    for tpl in info:
        x, y = tpl
        if x < nx: nx = x
        if x > mx: mx = x  
        if y < ny: ny = y 
        if y > my: my = y  
       
        
    return ((nx, mx), (ny, my))
        

def get_increments(vertices):
    steps = []
    for i in range(1,len(vertices)):
        prev = vertices[i-1]
        this = vertices[i]
        x1 = prev['x']; y1 = prev['y']
        x2 = this['x']; y2 = this['y']
        x_step = round(abs(x2 - x1)); y_step = round(abs(y2 - y1))
        
        step = (x_step, y_step)
        
        steps.append(step)
        
    return steps


def parse_poly(dict_item):
    label_class = dict_item['label_class']
    obj_id = dict_item['object_id']
    print('\tobject_id: {}'.format(obj_id))
    print('\t\tlabel_class: {}'.format(label_class))

    rinfo = []
    if 'vertices' in list(dict_item.keys()):
        vertices = dict_item['vertices']
        rinfo.append(list(get_increments(vertices)))
        
    elif 'regions' in list(dict_item.keys()):
        regions = dict_item['regions']
        for vertices in regions:
            rinfo.append(get_increments(vertices))
            
    num_regions = len(rinfo)
    print('\t\tNumber of regions: {}'.format(num_regions))
    for i, info in enumerate(rinfo):
        m, n = get_stats(info)
        
        print('\t\t\tregion: {}, n_vertices: {}'.format(i, len(info)+1))
        print('\t\t\tincrement stats: x(min, max): {}, y(min, max): {}'.format(n, m))
            
    return


def parse_annotation(label_folder, log_folder=None):
    
    json_file_list = list(sorted(os.listdir(label_folder)))
    
    if log_folder is not None:
        fullfile = os.path.join(log_folder, 'human.txt')
        h_file = open(fullfile, 'w')
        fullfile = os.path.join(log_folder, 'machine.txt')
        m_file = open(fullfile, 'w')

    for ann_file in json_file_list:
        fullfile = os.path.join(label_folder, ann_file)
        data_infos = load(fullfile)
        print('filename: {}'.format(ann_file))
        v_list = []; s_list = []
        for dict_item in data_infos:
            if 'label_type' in list(dict_item.keys()):
                if dict_item['label_type'] == 'polygon':
                    parse_poly(dict_item)

            

if __name__ == '__main__':
    image_prefix = './data/ruth/datasets/belt_data_natural/label_frames'
    annotation_prefix = './data/ruth/datasets/belt_data_natural/seg_labels_json'
    log_prefix = './data/belt_data_natural'
    
    for belt in BELTS:
        input_image_prefix = os.path.join(image_prefix, belt)
        input_annotation_prefix = os.path.join(annotation_prefix, belt)
        belt_prefix = os.path.join(log_prefix, belt)

        
        parse_annotation(input_annotation_prefix, log_prefix)