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
    
# def sub_processor(pid, result_dict, sub_file_list):
    # for idx, imgname in enumerate(sub_file_list):
        # annodict = annodicts[imgname]
        # height   = annodict['image size']['height']*scale
        # width    = annodict['image size']['width']*scale
        # image_id = annodict['image id']
        # objects  = annodict['preds']
        
        # visible_bodys = np.array(objects[1])
        # full_bodys    = np.array(objects[2])
        # head_boxs     = np.array(objects[3])
        # vehicles      = np.array(objects[4])
        
        # if(annmode == 'head'):
            # annmode_boxes = copy.deepcopy(head_boxs)*scale
        # if(annmode == 'full_body'):
            # annmode_boxes = copy.deepcopy(full_bodys)*scale
        # if(annmode == 'visible_body'):
            # annmode_boxes = copy.deepcopy(visible_bodys)*scale
        # if(annmode == 'vehicle'):
            # annmode_boxes = copy.deepcopy(vehicles)*scale
            
        # ### KMeans
        # points = (annmode_boxes[:, 0::2] + annmode_boxes[:, 1::2])/2.0
        # kmeans = KMeans(n_clusters = n_clusters, max_iter = 300, n_init = 10, init = 'k-means++', random_state = 0)
        # labels = kmeans.fit_predict(points)
        
        # ### cut 
        # image = cv2.imread(basepath + imgname)
        # image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        # print(image.shape)
        
        # for clu in range(n_clusters):
            # clu_fulls = annmode_boxes[labels == clu]
            
            # left   = min(clu_fulls[:, 0])
            # top    = min(clu_fulls[:, 1])
            # right  = max(clu_fulls[:, 2])
            # bottom = max(clu_fulls[:, 3])
            
            # if((bottom-top) <minside):
                # cnt = (bottom+top)/2.0
                # top = max(cnt - minside//2, 0)
                # bottom = min(cnt + minside//2, height)

            # if((right-left)<minside):
                # cnt   = (right+left)/2.0
                # left  = max(cnt - minside//2, 0)
                # right = min(cnt + minside//2, width)

            # boader_x = max(clu_fulls[:, 2]- clu_fulls[:, 0])//2
            # boader_y = max(clu_fulls[:, 3]- clu_fulls[:, 1])//2
            
            # top    = int(max(top-boader_y, 0))
            # left   = int(max(left-boader_x, 0))
            # bottom = int(min(bottom+boader_y, height))
            # right  = int(min(right+boader_x, width))
            
            # ious = computeIOUS(np.array([left, top, right, bottom]), annmode_boxes)
            # select_boxs = annmode_boxes[ious>=cutiou]
            # ious_boxs   = annmode_boxes[ious>0]
            # ious        = ious[ious>0]
            # ingore_boxs = ious_boxs[ious<cutiou]
            
            # subimg = copy.deepcopy(image[top: bottom, left: right])
            # select_boxs[:, 0] -= left
            # select_boxs[:, 1] -= top
            # select_boxs[:, 2] -= left
            # select_boxs[:, 3] -= top
            
            # # color = [random.randint(100,255), random.randint(100,255), random.randint(100,255)]
            # # cv2.rectangle(image, (int(left), int(top), int(right)-int(left)+1, int(bottom)-int(top)+1), color, 8, 16)
            
            # for box in select_boxs:        
                # cv2.rectangle(subimg, (int(box[0]), int(box[1]), int(box[2])-int(box[0])+1, int(box[3])-int(box[1])+1), [0,0,255], 8, 16)
            
            # save_name = imgname.replace('/', '_').split('.')[0] + '__' + str(scale) + '__' + str(left) + '__' + str(top) + '.jpg'
            # cv2.imwrite(outpath + save_name, subimg)
        
            # su_height, su_width = subimg.shape[:2]
            # result_dict[save_name] = {'image_id':image_id, 'scale':scale, 'left':left, 'top':top, 'width': su_width, 'height': su_height}
        # if(idx >2):
            # break

def sub_processor(pid, result_dict, sub_file_list):
    for idx, imgname in enumerate(sub_file_list):
        annodict = annodicts[imgname]
        height   = annodict['image size']['height']*scale
        width    = annodict['image size']['width']*scale
        image_id = annodict['image id']
        objects  = annodict['preds']
        
        visible_bodys = np.array(objects[1])
        full_bodys    = np.array(objects[2])
        head_boxs     = np.array(objects[3])
        vehicles      = np.array(objects[4])
        
        if(annmode == 'head'):
            annmode_boxes = copy.deepcopy(head_boxs)*scale
        if(annmode == 'full_body'):
            annmode_boxes = copy.deepcopy(full_bodys)*scale
        if(annmode == 'visible_body'):
            annmode_boxes = copy.deepcopy(visible_bodys)*scale
        if(annmode == 'vehicle'):
            annmode_boxes = copy.deepcopy(vehicles)*scale
            
        ### KMeans
        if(len(annmode_boxes) < 2.0*n_clusters and annmode == 'vehicle'):
            c_clusters = len(annmode_boxes)//4
        else:
            c_clusters = n_clusters
            
        points = (annmode_boxes[:, 0::2] + annmode_boxes[:, 1::2])/2.0
        kmeans = KMeans(n_clusters = int(c_clusters), max_iter = 300, n_init = 10, init = 'k-means++', random_state = 0)
        labels = kmeans.fit_predict(points)
        
        ### cut 
        image = cv2.imread(basepath + imgname)
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        print(image.shape)
        
        for clu in range(n_clusters):
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
            
            subimg = copy.deepcopy(image[top: bottom, left: right])
            
            ious = computeIOUS(np.array([left, top, right, bottom]), annmode_boxes)
            select_boxs = annmode_boxes[ious>=cutiou]
            ious_boxs   = annmode_boxes[ious>0]
            ious        = ious[ious>0]
            ingore_boxs = ious_boxs[ious<cutiou]
            select_boxs[:, 0] -= left
            select_boxs[:, 1] -= top
            select_boxs[:, 2] -= left
            select_boxs[:, 3] -= top
            
            if((right-left)>5120 or (bottom-top)>5120):
                height_p, width_p = subimg.shape[:2]
               # for left_p, top_p, right_p, bottom_p in zip([0, width_p//2], [0,height_p//2], [width_p//2, width_p], [height_p//2,height_p]):
                left_p, top_p = 0, 0
                while left_p < width_p:
                    if left_p + 3840 >= width_p:
                        left_p = max(width_p - 3840, 0)
                        
                    top_p = 0
                    while top_p < height_p:
                        if top_p + 3840 >= height_p:
                            top_p = max(height_p - 3840, 0)
                            
                        right_p  = min(left_p + 3840, width_p - 1)
                        bottom_p = min(top_p + 3840, height_p - 1)
                
                        subimg_p = copy.deepcopy(subimg[top_p: bottom_p, left_p: right_p])
                        
                        ious = computeIOUS(np.array([left_p, top_p, right_p, bottom_p]), select_boxs)
                        select_boxs_p = select_boxs[ious>=cutiou]
                        select_boxs_p[:, 0] -= left_p
                        select_boxs_p[:, 1] -= top_p
                        select_boxs_p[:, 2] -= left_p
                        select_boxs_p[:, 3] -= top_p
                        
                        # for box in select_boxs_p:        
                            # cv2.rectangle(subimg_p, (int(box[0]), int(box[1]), int(box[2])-int(box[0])+1, int(box[3])-int(box[1])+1), [0,0,255], 8, 16)
                            
                        save_name = imgname.replace('/', '_').split('.')[0] + '__' + str(scale) + '__' + str(left+left_p) + '__' + str(top+top_p) + '.jpg'
                        cv2.imwrite(outpath + save_name, subimg_p)
                    
                        su_height, su_width = subimg_p.shape[:2]
                        result_dict[save_name] = {'image_id':image_id, 'scale':scale, 'left':left+left_p, 'top':top+top_p, 'width': su_width, 'height': su_height}
                        
                        if top_p + 3840 >= height_p:
                            break
                        else:
                            top_p = top_p + 3000
                    if left_p + 3840 >= width_p:
                        break
                    else:
                        left_p = left_p + 3000 
            else:
                # for box in select_boxs:        
                    # cv2.rectangle(subimg, (int(box[0]), int(box[1]), int(box[2])-int(box[0])+1, int(box[3])-int(box[1])+1), [0,0,255], 8, 16)
                    
                save_name = imgname.replace('/', '_').split('.')[0] + '__' + str(scale) + '__' + str(left) + '__' + str(top) + '.jpg'
                cv2.imwrite(outpath + save_name, subimg)
            
                su_height, su_width = subimg.shape[:2]
                result_dict[save_name] = {'image_id':image_id, 'scale':scale, 'left':left, 'top':top, 'width': su_width, 'height': su_height}
            
if __name__ == '__main__':
    basepath  = '/data/Dataset/GigaVersion/image_test/'
    annofileF = '/data/Dataset/GigaVersion/image_annos/person_bbox_test.json'
    annofileR = '/data/Dataset/GigaVersion/Tsubmit.json'
    
    outpath     = '/data/Dataset/GigaVersion/image_test_split_k_means_20/image_visible2/'
    outannofile = '/data/Dataset/GigaVersion/image_test_split_k_means_20/test_visible2.json'

    pred_dicts = {}
    sub_data   = json.load(open(annofileR, 'r'))    
    for item in sub_data:
        img_id = item['image_id']
        category_id = item['category_id']
        bb_left   = item['bbox_left']
        bb_top    = item['bbox_top']
        bb_width  = item['bbox_width']
        bb_height = item['bbox_height']
        score     = item["score"]
        
        if(img_id not in pred_dicts.keys()):
            pred_dicts[img_id] = {}
            pred_dicts[img_id][1] = []
            pred_dicts[img_id][2] = []
            pred_dicts[img_id][3] = []
            pred_dicts[img_id][4] = []
        
        if(score>0.5):
            pred_dicts[img_id][category_id].append([bb_left, bb_top, bb_left+bb_width, bb_top+bb_height])
        
    annodicts = json.load(open(annofileF, 'r'))    
    imgnames  = [name for name in annodicts.keys()]
    
    for name in imgnames:
        img_id  = annodicts[name]['image id']
        pd_boxs = pred_dicts[img_id]
        annodicts[name]['preds'] = pd_boxs
    
    annmode    = 'visible_body'
    n_clusters = 25
    scale      = 0.5
    minside    = 1200
    cutiou     = 0.8

    ### multi_preocess 
    thread_num  = 32
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