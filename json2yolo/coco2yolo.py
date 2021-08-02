import os
import pycocotools.coco as coco
import numpy as np
import shutil
import json

def xywh2xxyy(box):
	x1 = box[0]
	y1 = box[1]
	x2 = box[0] + box[2]
	y2 = box[1] + box[3]
	return (x1, x2, y1, y2)

def convert(size, box):
	dw = 1. / (size[0])
	dh = 1. / (size[1])
	x = (box[0] + box[1]) / 2.0 - 1
	y = (box[2] + box[3]) / 2.0 - 1
	w = box[1] - box[0]
	h = box[3] - box[2]
	x = x * dw
	w = w * dw
	y = y * dh
	h = h * dh
	return (x, y, w, h)
    
if __name__ == '__main__':
    src_dir   = '/data/Dataset/GigaVersion/image_test_split_k_means_20/image_full2/'
    save_dir  = '/data/Dataset/GigaVersion/image_test_split_k_means_20/image_full2/'
    json_file = '/data/Dataset/GigaVersion/image_test_split_k_means_20/FULL2_COCO.json'
    coco_data = coco.COCO(json_file)
    images    = coco_data.getImgIds()
    
    data = {}
    for img_id in images:
        file_item = coco_data.loadImgs(ids=[img_id])[0]
        file_name = file_item['file_name']
        height, width = file_item['height'], file_item['width']
        
        ann_ids   = coco_data.getAnnIds(imgIds=[img_id])
        anns      = coco_data.loadAnns(ids=ann_ids)
        
        data[file_name] = list()
        for ann in anns:
            cation_id = ann['category_id']
            box = ann['bbox']

            box = xywh2xxyy(box)
            box = convert((width, height), box)
            
            label = '0 {} {} {} {}'.format(round(box[0],4), round(box[1],4), round(box[2],4), round(box[3],4))
            data[file_name].append(label)

    for file_name in data.keys():
        labels  = data[file_name]
        out_txt = save_dir + file_name.replace('.jpg', '.txt')
        
       # shutil.copyfile(src_dir + file_name, save_dir + file_name)
      #  
        fp = open(out_txt, 'w')
        for label in labels:
            fp.write(label + '\n')
        fp.close()
        