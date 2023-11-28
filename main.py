import cv2
import warnings
# 忽略特定类型的警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=Warning)  # WARNING
warnings.filterwarnings("ignore")
warnings.filterwarnings("always")
import numpy as np
import os,json
import math
import argparse
import matplotlib.pyplot as plt
plt.ion()
from PIL import Image
import torch
from torchvision import transforms
from yolo_face_detect_weights.yolo_v8_face import YOLOv8_face
import cv2
import mediapipe as mp
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import deepface 
from deepface import DeepFace
import cv2


models = [
"VGG-Face", 
"Facenet", 
"Facenet512", 
"OpenFace", 
"DeepFace", 
"DeepID", 
"ArcFace", 
"Dlib", 
"SFace",
]


def detect(cap):
    while True:
        # 逐帧捕获
        ret, img = cap.read()
        # 检测目标
        boxes, scores, classids, kpts = YOLOv8_face_detector.detect(img)
        # 未检测到人脸
        if len(boxes) == 0: 
            detect_flag = 0
            print('未 检 测 到 人 脸')
            text = 'WARNING! no people in screen!'
            cv2.putText(img, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            pass
        # 检测到人脸
        else:                
            detect_flag = 1
            print('当前有{}人'.format(len(boxes)))
            pts = np.array(boxes, dtype=np.int32)
            for i in pts:
                x, y, w, h = i  # [x,y,w,h]
                try:
                    roi = img[int(0.85*y):int(1.15*(y+h)), int(0.85*x):int(1.15*(x+w))]
                    # roi = cv2.resize(roi,(256,256))
                    cv2.imshow('face', roi)
                    cv2.waitKey(2)
                    # 识别目标
                    dfs = DeepFace.find(img_path = roi,db_path = "dataset", model_name = models[2],enforce_detection=False)
                    # print(dfs)
                    # 判断是谁
                    df = dfs[0]
                    df['ID'] = df['identity'].str.extract(r'\\(.*?)\/')
                    print(df,'\n')
                    n_decide = int(len(df['ID'])/5)
                    if n_decide>5:
                        n_decide = 5
                    # 找到'Facenet_cosine'列最大的n_decide所在的行
                    top_five_rows = df.nlargest(2, 'Facenet512_cosine')
                    # 提取这n_decide个行中的'ID'列的唯一值
                    unique_ids = top_five_rows['ID'].unique()
                    # 检查唯一的'ID'值的数量
                    if len(unique_ids) == 1 and df['Facenet512_cosine'].max() >= args.Facenet_cosine:
                        person_ID = unique_ids[0]
                        print("The IDs are the same:", person_ID,'\n')
                        
                        # 在数据库中
                        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)  # 绿色矩形框
                        # 在边界框上标注文字
                        cv2.putText(img, str(person_ID), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    else:
                        # 不在数据库中
                        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)  # 红色矩形框
                        # 在边界框上标注文字
                        cv2.putText(img, 'Not in DB', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                except:
                    pass
        # show
        winName = 'face_detection_and_recognize'
        cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(winName, 1000, 1000)
        cv2.imshow(winName, img)
        cv2.waitKey(2)




if __name__ == '__main__':
    #-------------------------------------------------------
    #-------------   参  数  配  置   ----------------------
    #-------------------------------------------------------
    parser = argparse.ArgumentParser()
    
    # 目标检测
    parser.add_argument('--imgpath', type=str, default='images_test/2.jpg', help="image path")
    parser.add_argument('--modelpath', type=str, default='./yolo_face_detect_weights/yolov8n-face.onnx',help="onnx filepath")
    parser.add_argument('--confThreshold', default=0.45, type=float, help='class confidence')
    parser.add_argument('--nmsThreshold', default=0.5, type=float, help='nms iou thresh')
    
    # 识别参数
    parser.add_argument('--Facenet_cosine', default=0.2, type=float, help='recognize Facenet_cosine Threshold')
    args = parser.parse_args()


    #-------------------------------------------------------
    #-------------   初     始    化   ----------------------
    #-------------------------------------------------------
    # 初始化YOLOv8_face 人脸检测器。
    YOLOv8_face_detector = YOLOv8_face(args.modelpath, conf_thres=args.confThreshold, iou_thres=args.nmsThreshold)


    #-------------------------------------------------------
    #-------------   摄 像 头 处 理   ----------------------
    #-------------------------------------------------------
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    detect(cap)

