import os
import cv2
import json
import copy
import random 
import numpy as np
import threading
import multiprocessing as mp

def sub_processor(pid, sub_file_list):
    for idx, imgname in enumerate(sub_file_list):
        print(imgname)
        annodict = annodicts[imgname]
        height   = annodict['image size']['height']*scale
        width    = annodict['image size']['width']*scale
        image_id = annodict['image id']
        objects  = annodict['preds']
        
        visible_bodys = np.array(objects[1])*scale
        full_bodys    = np.array(objects[2])*scale
        head_boxs     = np.array(objects[3])*scale
        vehicles      = np.array(objects[4])*scale
        
        print('img reading ')
        image = cv2.imread(basepath + imgname)
        print('img resize ')
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        print('img draw ')
        for box in visible_bodys:        
            cv2.rectangle(image, (int(box[0]), int(box[1]), int(box[2])-int(box[0])+1, int(box[3])-int(box[1])+1), [0,0,255], 8, 16)

        for box in full_bodys:        
            cv2.rectangle(image, (int(box[0]), int(box[1]), int(box[2])-int(box[0])+1, int(box[3])-int(box[1])+1), [0,0,255], 8, 16)
        
        for box in head_boxs:        
            cv2.rectangle(image, (int(box[0]), int(box[1]), int(box[2])-int(box[0])+1, int(box[3])-int(box[1])+1), [0,0,255], 8, 16)

        for box in vehicles:        
            cv2.rectangle(image, (int(box[0]), int(box[1]), int(box[2])-int(box[0])+1, int(box[3])-int(box[1])+1), [0,0,255], 8, 16)

        print('img write ')
        cv2.imwrite('Vis/' + imgname.split('/')[-1], image)
 
if __name__ == '__main__':
    basepath  = '/data/Dataset/GigaVersion/image_test/'
    annofileF = '/data/Dataset/GigaVersion/image_annos/person_bbox_test.json'
    annofileR = '/data/Dataset/GigaVersion/image_test_split_k_means_20/results/vehicle_wbf.json'
    
    pred_dicts = {}
    sub_data  = json.load(open(annofileR, 'r'))    
    for item in sub_data:
        img_id = item['image_id']
        category_id = item['category_id']
        bb_left   = item['bbox_left']
        bb_top    = item['bbox_top']
        bb_width  = item['bbox_width']
        bb_height = item['bbox_height']

        if(img_id not in pred_dicts.keys()):
            pred_dicts[img_id] = {}
            pred_dicts[img_id][1] = []
            pred_dicts[img_id][2] = []
            pred_dicts[img_id][3] = []
            pred_dicts[img_id][4] = []
        
        pred_dicts[img_id][category_id].append([bb_left, bb_top, bb_left+bb_width, bb_top+bb_height])
        
    annodicts = json.load(open(annofileF, 'r'))    
    imgnames  = [name for name in annodicts.keys()]
    random.shuffle(imgnames)
    
    print('start ann')
    for name in imgnames:
        img_id  = annodicts[name]['image id']
        pd_boxs = pred_dicts[img_id]
        annodicts[name]['preds'] = pd_boxs
    
    annmode    = 'all'
    scale      = 0.2

    print('start drawing')
    sub_processor(0, imgnames)

# if __name__ == '__main__':
    # basepath  = '/data/Dataset/GigaVersion/image_test/'
    # annofileF = '/data/Dataset/GigaVersion/image_annos/person_bbox_test.json'
    # annofileR = '/data/Dataset/GigaVersion/image_test_split_k_means_20/yolov5l6_submit.json'
    
    # pred_dicts = {}
    # sub_data  = json.load(open(annofileR, 'r'))    
    # for item in sub_data:
        # img_id = item['image_id']
        # category_id = item['category_id']
        # bb_left   = item['bbox_left']
        # bb_top    = item['bbox_top']
        # bb_width  = item['bbox_width']
        # bb_height = item['bbox_height']

        # if(img_id not in pred_dicts.keys()):
            # pred_dicts[img_id] = {}
            # pred_dicts[img_id][1] = []
            # pred_dicts[img_id][2] = []
            # pred_dicts[img_id][3] = []
            # pred_dicts[img_id][4] = []
        
        # pred_dicts[img_id][category_id].append([bb_left, bb_top, bb_left+bb_width, bb_top+bb_height])
        
    # annodicts = json.load(open(annofileF, 'r'))    
    # imgnames  = [name for name in annodicts.keys()]
    # random.shuffle(imgnames)
    
    # for name in imgnames:
        # img_id  = annodicts[name]['image id']
        # pd_boxs = pred_dicts[img_id]
        # annodicts[name]['preds'] = pd_boxs
    
    # annmode    = 'all'
    # scale      = 0.5

    # ### multi_preocess 
    # thread_num = 6
    # per_thread_file_num = len(imgnames) // thread_num

    # print('start drawing')
    # processes = []
    # for pid in range(thread_num):
        # if pid == thread_num - 1:
            # sub_file_list = imgnames[pid * per_thread_file_num:]
        # else:
            # sub_file_list = imgnames[pid * per_thread_file_num: (pid + 1) * per_thread_file_num]
        # p = mp.Process(target=sub_processor, args=(pid, sub_file_list))
        # p.start()
        # processes.append(p)

    # for p in processes:
        # p.join()
        