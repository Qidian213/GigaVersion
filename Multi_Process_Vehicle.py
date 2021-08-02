import os
import cv2
import json
import copy
import random 
import numpy as np
import threading
import multiprocessing as mp
from sklearn.cluster import KMeans

def computeIOU(rec1, rec2):
    cx1, cy1, cx2, cy2 = rec1
    gx1, gy1, gx2, gy2 = rec2
    S_rec2 = (gx2 - gx1 + 1) * (gy2 - gy1 + 1)
    x1 = max(cx1, gx1)
    y1 = max(cy1, gy1)
    x2 = min(cx2, gx2)
    y2 = min(cy2, gy2)
 
    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)
    area = w * h
    iou = area/S_rec2
    return iou
    
def computeIOUS(rect, bboxs, iou=0.5):
    ious = []
    for box in bboxs:
        iou = computeIOU(rect, box)
        ious.append(iou)
        
    return np.array(ious)
    
def sub_processor(pid, result_dict, sub_file_list):
    for idx, imgname in enumerate(sub_file_list):
        try:
            annodict = annodicts[imgname]
            height   = annodict['image size']['height']*scale
            width    = annodict['image size']['width']*scale
            
            objects = annodict['objects list']
            
            ignores    = []
            fakes      = []
            vehicles   = []
            valid_anns = []
            for obj in objects:
                if(obj['category'] == 'fake'):
                    rect = obj['rect']
                    x1, y1, x2, y2 = rect['tl']['x']*width, rect['tl']['y']*height, rect['br']['x']*width, rect['br']['y']*height
                    fakes.append([x1, y1, x2, y2])
                    
                elif(obj['category'] == 'ignore'):
                    rect = obj['rect']
                    x1, y1, x2, y2 = rect['tl']['x']*width, rect['tl']['y']*height, rect['br']['x']*width, rect['br']['y']*height
                    ignores.append([x1, y1, x2, y2])

                elif(obj['category'] == 'vehicles'):
                    rect = obj['rect']
                    x1, y1, x2, y2 = rect['tl']['x']*width, rect['tl']['y']*height, rect['br']['x']*width, rect['br']['y']*height
                    vehicles.append([x1, y1, x2, y2])
                    
                else:
                    rect = obj['rect']
                    x1, y1, x2, y2 = rect['tl']['x']*width, rect['tl']['y']*height, rect['br']['x']*width, rect['br']['y']*height
                    valid_anns.append([x1, y1, x2, y2])

            ignores    = np.array(ignores)
            fakes      = np.array(fakes)
            vehicles   = np.array(vehicles)
            valid_anns = np.array(valid_anns)

            annmode_boxes = copy.deepcopy(valid_anns)
                
            ### KMeans
            if(len(annmode_boxes) < 2.0*n_clusters):
                c_clusters = len(annmode_boxes)//2
            else:
                c_clusters = n_clusters
                
            points = (annmode_boxes[:, 0::2] + annmode_boxes[:, 1::2])/2.0
            kmeans = KMeans(n_clusters = int(c_clusters), max_iter = 300, n_init = 10, init = 'k-means++', random_state = 0)
            labels = kmeans.fit_predict(points)
            
            ### cut 
            image = cv2.imread(basepath + imgname)
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            
            for box in fakes:   
                image[int(box[1]): int(box[3]), int(box[0]):int(box[2])] = np.array([125,125,125])
            for box in ignores:   
                image[int(box[1]): int(box[3]), int(box[0]):int(box[2])] = np.array([125,125,125])
            for box in vehicles:   
                image[int(box[1]): int(box[3]), int(box[0]):int(box[2])] = np.array([125,125,125])
                
            for clu in range(c_clusters):
                clu_fulls = annmode_boxes[labels == clu]
                
                left   = min(clu_fulls[:, 0])
                top    = min(clu_fulls[:, 1])
                right  = max(clu_fulls[:, 2])
                bottom = max(clu_fulls[:, 3])
                
                if((bottom-top) <minside):
                    cnt = (bottom+top)/2.0
                    top = max(cnt - minside//2, 0)
                    bottom = min(cnt + minside//2, height)

                if((right-left)<minside):
                    cnt   = (right+left)/2.0
                    left  = max(cnt - minside//2, 0)
                    right = min(cnt + minside//2, width)

                boader_x = max(clu_fulls[:, 2]- clu_fulls[:, 0])//2
                boader_y = max(clu_fulls[:, 3]- clu_fulls[:, 1])//2
                
                top    = int(max(top-boader_y, 0))
                left   = int(max(left-boader_x, 0))
                bottom = int(min(bottom+boader_y, height))
                right  = int(min(right+boader_x, width))
                
                ious = computeIOUS(np.array([left, top, right, bottom]), annmode_boxes)
                select_boxs = annmode_boxes[ious>=cutiou]
                ious_boxs   = annmode_boxes[ious>0]
                ious        = ious[ious>0]
                ingore_boxs = ious_boxs[ious<cutiou]
                
                subimg = copy.deepcopy(image[top: bottom, left: right])
                select_boxs[:, 0] -= left
                select_boxs[:, 1] -= top
                select_boxs[:, 2] -= left
                select_boxs[:, 3] -= top
                
                ingore_boxs[:, 0] -= left
                ingore_boxs[:, 1] -= top
                ingore_boxs[:, 2] -= left
                ingore_boxs[:, 3] -= top
                
                # color = [random.randint(100,255), random.randint(100,255), random.randint(100,255)]
                # cv2.rectangle(image, (int(left), int(top), int(right)-int(left)+1, int(bottom)-int(top)+1), color, 8, 16)
                
                for box in ingore_boxs:   
                    subimg[int(box[1]): int(box[3]), int(box[0]):int(box[2])] = np.array([125,125,125])
                
                # for box in select_boxs:        
                    # cv2.rectangle(subimg, (int(box[0]), int(box[1]), int(box[2])-int(box[0])+1, int(box[3])-int(box[1])+1), [0,0,255], 8, 16)
                
                save_name = imgname.replace('/', '_').split('.')[0] + '__' + str(scale) + '__' + str(left) + '__' + str(top) + '.jpg'
                cv2.imwrite(outpath + save_name, subimg)
            
                su_height, su_width = subimg.shape[:2]
                result_dict[save_name] = {'width': su_width, 'height': su_height, 'bboxs': select_boxs.tolist()}
        except:
            pass
            
if __name__ == '__main__':
    basepath = '/data/Dataset/GigaVersion/image_train/'
    annofile = '/data/Dataset/GigaVersion/image_annos/vehicle_bbox_train.json'
    
    outpath     = '/data/Dataset/GigaVersion/image_vehicle_split_k_means_20/image_train/'
    outannofile = '/data/Dataset/GigaVersion/image_vehicle_split_k_means_20/panda_vehicle.json'
    
    annodicts = json.load(open(annofile, 'r'))    
    imgnames  = [name for name in annodicts.keys()]
    random.shuffle(imgnames)
    
    annmode    = 'vehicle'
    n_clusters = 15
    scale      = 0.5
    minside    = 1200
    cutiou     = 0.8

    ### multi_preocess 
    thread_num  = 24
    per_thread_file_num = len(imgnames) // thread_num
    result_dict = mp.Manager().dict()
    
    processes = []
    for pid in range(thread_num):
        if pid == thread_num - 1:
            sub_file_list = imgnames[pid * per_thread_file_num:]
        else:
            sub_file_list = imgnames[pid * per_thread_file_num: (pid + 1) * per_thread_file_num]
        p = mp.Process(target=sub_processor, args=(pid, result_dict, sub_file_list))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        
    fp = open(outannofile, 'w')
    json.dump(dict(result_dict), fp)
    