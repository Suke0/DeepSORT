# -*- coding: utf-8 -*-
# @Time    : 2020/6/10 14:36
# @Author  : Suke0
# @Email   : 652434288@qq.com
# @File    : main.py
# @Software: PyCharm
from collections import deque

import cv2
import time
import numpy as np
from PIL import Image

from deep_sort.detection import Detection
from deep_sort.iou_matching import non_max_suppression
from deep_sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort.tracker import Tracker
from re_id.generator_detection import box_encoder
from re_id.reid_model import reid_model
from yolo_v3.yolo import YOLOV3
#import tensorflow as tf

from yolo_v3.yolo import YOLOV3


np.random.seed(100)
COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")
pts = [deque(maxlen=30) for _ in range(9999)]
def main(yolov3):
    start = time.time()
    max_cosine_distance = 0.5
    nn_budget = None
    nms_max_overlap = 0.3

    counter = []
    model_filename = './re_id/reid_weight.h5'
    encoder = box_encoder(model_filename)
    #re_id_model.load_weights(model_filename)
    metric = NearestNeighborDistanceMetric('cosine',max_cosine_distance,nn_budget)
    tracker = Tracker(metric)

    #writeVideo_flag = True
    video_capture = cv2.VideoCapture("./test_video/MOT16-08-raw.webm")
    # Define the codec and create VideoWriter object
    w = int(video_capture.get(3))
    h = int(video_capture.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('./output/0_output.avi', fourcc, 15, (w, h))
    list_file = open('detection.txt','w')
    frame_index = -1

    fps = 0.0

    while True:
        ret, frame = video_capture.read() # frame shape h*w*3
        if not ret:
            break
            pass
        t1 = time.time()
        image = Image.fromarray(frame[...,::-1]) #bgr to rgb w*h
        t_detect_image = time.time()
        boxs, class_names = yolov3.detect_image(image)
        t_detect_image_ = time.time()
        print("t_detect_image"+str(t_detect_image_ - t_detect_image))
        t_detect_image = time.time()
        features = encoder(frame, boxs)
        t_detect_image_ = time.time()
        print("t_box_encoder" + str(t_detect_image_ - t_detect_image))
        # t_detect_image0
        # 0.4851090908050537
        # t_box_encoder0
        # 0.03600788116455078
        # score to 1.0 here.
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs,features)]
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d  in detections])
        scores = np.array([d.confidence for d in detections])
        indices = non_max_suppression(boxes, nms_max_overlap,scores)
        detections = [detections[i] for i in indices]

        tracker.predict()
        tracker.update(detections)

        i = 0
        indexIDs = []
        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
            pass

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
                pass

            indexIDs.append(int(track.track_id))
            counter.append(int(track.track_id))
            bbox = track.to_tlbr()
            color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]

            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (color), 3)
            cv2.putText(frame, str(track.track_id),(int(bbox[0]),int(bbox[1]-50)),0,5e-3*150,(color), 3)

            if len(class_names) > 0:
                class_names = class_names[0]
                cv2.putText(frame, str(class_names[0]), (int(bbox[0]), int(bbox[1] - 20)), 0, 5e-3 * 150, (color), 2)
                pass

            i += 1
            center = (int((bbox[0]+bbox[2])/2),int((bbox[1]+bbox[3])/2))
            # track_id[center]
            pts[track.track_id].append(center)
            thickness = 5
            cv2.circle(frame, center, 1, color, thickness)

            for j in range(1, len(pts[track.track_id])):
                if pts[track.track_id][j-1] is None or pts[track.track_id][j] is None:
                    continue
                    pass
                thickness = int(np.sqrt(64/float(j+1)) * 2)
                cv2.line(frame, (pts[track.track_id][j-1]),(pts[track.track_id][j]),(color),thickness)
                pass
            pass

        count = len(set(counter))
        cv2.putText(frame, "Total Object Counter: " + str(count), (int(20), int(120)), 0, 5e-3 * 200, (0, 255, 0), 2)
        cv2.putText(frame, "Current Object Counter: " + str(i), (int(20), int(80)), 0, 5e-3 * 200, (0, 255, 0), 2)
        cv2.putText(frame, "FPS: %f" % (fps), (int(20), int(40)), 0, 5e-3 * 200, (0, 255, 0), 3)
        cv2.namedWindow("YOLOV3_DeepSORT",0)
        cv2.resizeWindow("YOLOV3_DeepSORT", 1024, 768)
        cv2.imshow("YOLOV3_DeepSORT", frame)

        # save a frame
        out.write(frame)
        frame_index = frame_index + 1
        list_file.write(str(frame_index)+" ")
        if len(boxs) != 0:
            for i in range(0, len(boxs)):
                list_file.write(str(boxs[i][0]) + ' ' + str(boxs[i][1]) + ' ' + str(boxs[i][2]) + ' ' + str(boxs[i][3]) + ' ')
                pass
            pass
        list_file.write('\n')

        fps = (fps + (1./(time.time()-t1))) / 2

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            pass
        pass

    print("[Finish]")
    end = time.time()

    video_capture.release()
    #out.release()
    list_file.close()
    cv2.destroyAllWindows()
    pass

if __name__ == '__main__':
    main(YOLOV3())