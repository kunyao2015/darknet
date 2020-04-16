from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import sys
#sys.path.append('../')
import darknet
#from tracker import Tracker
from sort import *

def convertBack(x, y, w, h):
    xmin = max(int(round(x - (w / 2))),0)
    xmax = int(round(x + (w / 2)))
    ymin = max(int(round(y - (h / 2))),0)
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img, point=(0,0)):
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (point[0]+xmin, point[1]+ymin)
        pt2 = (point[0]+xmax, point[1]+ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        cv2.putText(img,
                    detection[0].decode() +
                    " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
    return img


def cal_iou(box1,box2):
    x1 = max(box1[0],box2[0])
    y1 = max(box1[1],box2[1])
    x2 = min(box1[2],box2[2])
    y2 = min(box1[3],box2[3])
    i = max(0,(x2-x1))*max(0,(y2-y1))
    u = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) -  i
    iou = float(i)/float(u)
    return iou

def get_objNameAndProb(item,objects):
    iou_list = []
    for i,object in enumerate(objects):
        x, y, w, h = object[2]
        x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
        iou_list.append(cal_iou(item[:4],[x1,y1,x2,y2]))
    max_index = iou_list.index(max(iou_list))
    return objects[max_index][0],objects[max_index][1]

netMain = None
metaMain = None
altNames = None


def YOLO():

    global metaMain, netMain, altNames
    configPath = "./cfg/yolov3.cfg"
    weightPath = "./yolov3.weights"
    metaPath = "./cfg/coco.data"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    print(netMain)
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass

    #lpnet, lpmeta = initDetect()

     # Create Object Tracker
    #tracker = Tracker(100, 30, 5, 100)
    mot_tracker = Sort(max_age = 10,min_hits = 2)

    # Variables initialization
    skip_frame_count = 0
    track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                    (0, 255, 255), (255, 0, 255), (255, 127, 255),
                    (127, 0, 255), (127, 0, 127)]


    #img_path = '/home/aipc/dataset/images/allImages/images/60698->22412266-100000166-1840.jpg'
    cap = cv2.VideoCapture("/home/aipc/code-YK/datasets/jiegouhua.avi")
    
    ret, frame_read = cap.read()
    frame_height, frame_width, _ = frame_read.shape
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("FPS:"+str(fps))
    out =  cv2.VideoWriter("/home/aipc/code-YK/datasets/jiegouhua_yolov3sort.avi", cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

    print("Starting the YOLO loop...")
    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(frame_width, frame_height,3)
    while True:
        ret, frame_read = cap.read()
        if not ret: break
        frame_height, frame_width, _ = frame_read.shape
        

        # Skip initial frames that display logo
        if (skip_frame_count < 15):
            skip_frame_count += 1
            continue

        img = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        darknet.copy_image_from_bytes(darknet_image,img.tobytes())

        prev_time = time.time()
        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.5)
        detections = [r for r in detections if r[0] in [b'person',b'bicycle',b'motorbike',b'car',b'bus',b'truck']]
        detections = np.array(detections)
        #print(detections)
        trackers = []
        if len(detections) != 0:
            class_name = detections[:,0][np.newaxis,:]  
            class_name = class_name.T
            prob = detections[:,1][np.newaxis,:]
            #print(prob)
            #print(prob.shape)
            prob = prob.T
            #print(prob.shape)
            detections_boxes = []
            for i in range(detections.shape[0]):
                if detections[i][2][0] < 0:
                    detections[i][2][0] = 0
                if detections[i][2][1] < 0:
                    detections[i][2][1] = 0
                x, y, w, h = detections[i][2][0],detections[i][2][1],detections[i][2][2],detections[i][2][3]
                xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h))
                box = [xmin,ymin,xmax,ymax]
                detections_boxes.append(box)
            #print(detections_boxes)
            #print(np.array(detections_boxes).shape)
            dets = np.hstack((np.array(detections_boxes),prob))
            #objects = np.hstack((class_name,dets))
            #print(dets)
            trackers = mot_tracker.update(dets)
        #print(trackers)
        #print(detections.shape[0],len(trackers))
        for i in range(len(trackers)):
            #print(str(len(trackers)) + ":"+str(i))
            #if (prob[i][0] < 0.75):
            #    continue
            #x0, y0, x1, y1 = boxes_1[i]
            #print(detections[i][0])
            objectName,p = get_objNameAndProb(trackers[i], detections)
            #print(objectName,prob)
            #print(trackers[i])
            x0, y0, x1, y1 = trackers[i][:4]
            clr = int(trackers[i][4]) % 9
            pt1 = (int(x0), int(y0))
            pt2 = (int(x1), int(y1))
            cv2.rectangle(frame_read, pt1, pt2, track_colors[clr], 2)
            cv2.putText(frame_read, str(objectName,encoding='utf-8') + " " + "ID:"  + str(int(trackers[i][4])) + " " + "prob:"+str(round(p,2)),
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    track_colors[clr], 2)
            
        '''
        centers = []
        for detection in detections:
            x, y, w, h = detection[2][0],\
                    detection[2][1],\
                    detection[2][2],\
                    detection[2][3]
            xmin, ymin, xmax, ymax = convertBack(
                float(x), float(y), float(w), float(h))
            cx = (xmin + xmax)//2
            cy = (ymin + ymax)//2
            centers.append((cx,cy))
            #print(cx,cy)

            #print(xmin, ymin, xmax, ymax)
            car_img = img[ymin:ymax,xmin:xmax,:]
            darknet_car_image = darknet.make_image(xmax-xmin, ymax-ymin,3)
            darknet.copy_image_from_bytes(darknet_car_image,car_img.tobytes())

            temp_time = time.time()
            lpdetections = darknet.detect_image(lpnet, lpmeta, darknet_car_image, thresh=0.5)
            #print(1000*(time.time()-temp_time))
            #img = cvDrawBoxes(lpdetections, img, point=(xmin,ymin))
            # cv2.imshow('car_img', car_img)
            # cv2.waitKey(5000)
        '''

        '''
        image = cvDrawBoxes(detections, img)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(1000*(time.time()-prev_time))
        if (len(centers) > 0):

            # Track object using Kalman Filter
            tracker.Update(centers)

            # For identified object tracks draw tracking line
            # Use various colors to indicate different track_id
            for i in range(len(tracker.tracks)):
                if (len(tracker.tracks[i].trace) > 1):
                    for j in range(len(tracker.tracks[i].trace)-1):
                        # Draw trace line
                        x1 = tracker.tracks[i].trace[j][0][0]
                        y1 = tracker.tracks[i].trace[j][1][0]
                        x2 = tracker.tracks[i].trace[j+1][0][0]
                        y2 = tracker.tracks[i].trace[j+1][1][0]
                        clr = tracker.tracks[i].track_id % 9
                        #cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)),
                        #         track_colors[clr], 2)
        '''
        cv2.imshow('Demo', frame_read)
        out.write(frame_read)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    

    '''
    cap.set(3, 1280)
    cap.set(4, 720)
    out = cv2.VideoWriter(
        "output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0,
        (darknet.network_width(netMain), darknet.network_height(netMain)))
    print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)
    while True:
        prev_time = time.time()
        ret, frame_read = cap.read()
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (darknet.network_width(netMain),
                                    darknet.network_height(netMain)),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())

        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)
        image = cvDrawBoxes(detections, frame_resized)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(1/(time.time()-prev_time))
        cv2.imshow('Demo', image)
        cv2.waitKey(3)
    
    cap.release()
    out.release()
    '''
if __name__ == "__main__":
    YOLO()
