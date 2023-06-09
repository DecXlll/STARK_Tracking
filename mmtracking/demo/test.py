import cv2
import numpy as np
def init_video_capture():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, int(60))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 360)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,0)
    print("fps:",fps)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    return cap,fps



cap,fps= init_video_capture()
s = np.array([0,0,0,0]) 
def show_xy(event,x,y,flags,userdata):
    s[0] = event
    s[1] = x
    s[2] = y
    s[3] = flags
    # 印出相關參數的數值，userdata 可透過 setMouseCallback 第三個參數垂遞給函式
init_bbox = np.array([0,0,0,0])
x1=0
y1=0
x2=0
y2=0
while(1):
    ret,img = cap.read()
    cv2.imshow('oxxostudio', img)
    cv2.setMouseCallback('oxxostudio', show_xy)  # 設定偵測事件的函式與視窗
    if(s[0]==1&s[3]==1):
        if(x1==0 & y1==0):
            print("1")
            x1 = s[1] 
            y1 = s[2]
    if(s[0]==4):
        print("2")
        x2 = s[1]
        y2 = s[2]
        init_bbox[0] = x1
        init_bbox[1] = y1
        init_bbox[2] = x2
        init_bbox[3] = y2
        
        x1=0;y1=0;x2=0;y2=0
        
        
    
    print(init_bbox,x1,y1,x2,y2,s)
    # print(s)
    cv2.waitKey(1)     # 按下任意鍵停止