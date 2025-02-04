import cv2
import time
import os
import argparse
import numpy as np

import rclpy
from rclpy.node import Node


class Dnn():
    def __init__(self,nms=0.4):

        self.modelfolder = os.path.join(os.getenv('HOME'),'av/models')
        self.datafolder = os.path.join(os.getenv('HOME'),'av/data')

        self.nms_threshold = 0.4
        self.NMS = nms

        self.COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

        self.class_names = []
        with open(os.path.join(self.modelfolder,"coco.names"), "r") as f:
                self.class_names = [cname.strip() for cname in f.readlines()]

        self.net = cv2.dnn.readNet(os.path.join(self.modelfolder,"yolov4-mish-416.weights"), os.path.join(self.modelfolder,"yolov4-mish-416.cfg"))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

        self.model = cv2.dnn_DetectionModel(self.net)
        self.model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

        self.run_time = 0


    def detect(self,frame): 
        start = time.time()
        classes, scores, boxes = self.model.detect(frame, self.nms_threshold, self.NMS)
        end = time.time()
        self.run_time = end - start
        return classes, scores, boxes



    def draw(self, frame, classes, scores, boxes, c_flag):

        bb_area_list = []

        start_drawing = time.time()
        for (classid, score, box) in zip(classes, scores, boxes): 
                color = self.COLORS[int(classid) % len(self.COLORS)]
                label = "%s : %f" % (self.class_names[classid], score)
                cv2.rectangle(frame, box, color, 2)
                box_area = abs((box[0] - box[2])*(box[1] - box[3]))
                bb_area_list.append(box_area)

                cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        end_drawing = time.time()
        fps_label = "FPS: %.2f (excluding drawing time of %.2fms)" % (1 / (self.run_time), (end_drawing - start_drawing) * 1000)
        cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        #print("start")
        #print(bb_area_list)

        caution_label = "Lane switch : No Vehicle in proximity"

        for b in bb_area_list:
                if (b > 350000):
                        caution_label = "Lane switch CAUTION : Vehicle proximity detected"

        cv2.putText(frame, caution_label, (0, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (10, 10, 255), 2) # was (700,800)

        if c_flag == 1:
                cv2.putText(frame, "Lane switch CAUTION : switch prohibited", (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (10, 10, 255), 2)# was (700,750)
        else :
                cv2.putText(frame, "Lane switch : permitted", (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (10, 10, 255), 2)# was (700,750)


        cv2.imshow("detections", frame)
        return frame


if __name__ == "__main__":
    dnn_mod = Dnn()
    parser = argparse.ArgumentParser(description = 'input video .mp4')
    parser.add_argument('vid', help='provide name your of video file. example.mp4')
    args = parser.parse_args()

    vid_path = os.path.join(self.datafolder,"msu_bags/"+ str(args.vid))
    vc = cv2.VideoCapture(vid_path)
    stop_key = cv2.waitKey(1)

    while stop_key < 1:
        if (stop_key == ord('q')) or (stop_key == ord('Q')):
            exit()
        (grabbed, frame) = vc.read()
        if not grabbed:
            exit()
        classes, scores, boxes = dnn_mod.detect(frame)
        dnn_mod.draw(frame, classes, scores, boxes)
        stop_key = cv2.waitKey(1)



