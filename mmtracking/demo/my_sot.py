# Copyright (c) OpenMMLab. All rights reserved.
from bdb import GENERATOR_AND_COROUTINE_FLAGS
import os
import os.path as osp
from sys import flags
import tempfile
from argparse import ArgumentParser
from time import sleep
import threading
import numpy as np
import copy

import cv2
import mmcv

import time

from mmtrack.apis import inference_sot, init_model

def get_Argument():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('--input', help='input video file')
    parser.add_argument('--output', help='output video file (mp4 format)')
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--show',action='store_true',default=False,help='whether to show visualizations.')
    parser.add_argument('--color', default=(0, 255, 0), help='Color of tracked bbox lines.')
    parser.add_argument('--thickness', default=3, type=int, help='Thickness of bbox lines.')
    parser.add_argument('--fps', type=int, help='FPS of the output video')
    parser.add_argument('--gt_bbox_file', help='The path of gt_bbox file')
    args = parser.parse_args()
    return args

def init_video_capture():
    args = get_Argument()
    #cap = cv2.VideoCapture("demo\car.mp4")
    cap = cv2.VideoCapture(-1)
    cap.set(cv2.CAP_PROP_FPS, int(args.fps))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,5)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    print(f"fps:{fps} ,WIDTH:{w},HEIGHT:{h}")
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    return cap,fps

def get_init_bbox(img):
    init_bbox = list(cv2.selectROI('selectROI', img, True, False))
    cv2.destroyWindow('selectROI')
    # convert (x1, y1, w, h) to (x1, y1, x2, y2)
    init_bbox[2] += init_bbox[0]
    init_bbox[3] += init_bbox[1]
    return init_bbox

def get_xy(event,x,y,flags,userdata):
    s[0] = event
    s[1] = x
    s[2] = y
    s[3] = flags
# img = np.zeros((360, 240), dtype = "uint8")

def get_mouse():
    x1=0;y1=0;x2=0;y2=0
    flag = 0
    while(1):
        cv2.imshow('box_select', img)
        cv2.setMouseCallback('box_select',get_xy) 
        cv2.waitKey(10) 
        if(s[0]==1 and s[3]==1):
            if(flag==0):
                x1 = s[1] 
                y1 = s[2]
                flag = 1
                
        elif(s[0]==4):
            # sleep(0.1)
            x2 = s[1]
            y2 = s[2]
            # print("x",x1,"y",y1)
            bbox[0] = x1
            bbox[1] = y1
            bbox[2] = x2
            bbox[3] = y2  
            flag = 0          
            # x1=0;y1=0
            # x2=0;y2=0
        ret=cv2.waitKey(1)   
        closed = cv2.getWindowProperty('box_select', cv2.WND_PROP_VISIBLE) < 1
        # if user closed window or if some key pressed
        if closed:
            break
        
        # print("bbox:",bbox,"   s:",s)
        # sleep(0.1)

def main():
    
    args = get_Argument()
    cap,fps = init_video_capture()
    # build the model from a config file and a checkpoint file
    print("loading model ...")
    model = init_model(args.config, args.checkpoint, device=args.device)
    global bbox,s,img
    no_bboxes = {'track_bboxes':np.array([0.,0.,0.,0.,0.])}
    s = np.array([0,0,0,0]) 
    i = 0;  t = 0.0;  total = 0.0 ;flag_1 = 0
    ret,img = cap.read()
    bbox = get_init_bbox(img)
    bboxx = bbox.copy()
    print(bbox)
    # init_bbox = {'track_bboxes':np.array([0.,0.,0.,0.,0.])}
    #Thread_mouse = threading.Thread(target=get_mouse)
    while(1):  
        ret,img = cap.read()
        # if(cv2.waitKey(1)==27): #press "b" to change bbox
        #     # init_bbox = get_init_bbox(img)
        #     # i=0  
        #     print("get_init_bbox")
        #     cv2.destroyAllWindows()
        #     sleep(3)
            # init_bbox = get_init_bbox(img)
        #     i=0 
        if(bbox==bboxx):
            start = time.perf_counter_ns()
            result = inference_sot(model, img, bboxx, frame_id=i)
            end = time.perf_counter_ns()
            total = total + (end - start)
            i = i+1
            t = t+1
            if t==30:
                print("process_time:",(total/30)/1000/1000,"ms")
                t=0
                total = 0
#             print("result:",result)

            
            if(result['track_bboxes'][4] >= 0.5):
                model.show_result(img,
                    result,
                    show=args.show,
                    win_name='show_result',
                    wait_time=int(1000. / fps) if fps else 0,
                    out_file=None,
                    thickness=args.thickness)
            elif(result['track_bboxes'][4] == -1.):
                model.show_result(img,
                    result,
                    show=args.show,
                    win_name='show_result',
                    wait_time=int(1000. / fps) if fps else 0,
                    out_file=None,
                    thickness=args.thickness)
            else:
                model.show_result(img,
                    no_bboxes,
                    show=args.show,
                    win_name='show_result',
                    wait_time=int(1000. / fps) if fps else 0,
                    out_file=None,
                    thickness=args.thickness)
        else:
            x_d = bbox[2]-bbox[0]
            y_d = bbox[3]-bbox[1]
            # print(x_d,y_d)
            if(x_d>=10 and y_d>=10):
                i = 0
                bboxx = copy.deepcopy(bbox)
            else:
                bbox = copy.deepcopy(bboxx)
        #if(flag_1==0):
        #    Thread_mouse.start()
        #    flag_1 = 1
        closed = cv2.getWindowProperty('show_result', cv2.WND_PROP_VISIBLE) < 1
        ## if user closed window or if some key pressed
        if closed :
            break
if __name__ == '__main__':
    main()
