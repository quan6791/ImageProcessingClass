#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
from timeit import time
import warnings
import sys
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
warnings.filterwarnings('ignore')


import base64
import cv2
import zmq

context = zmq.Context()
footage_socket = context.socket(zmq.PUB)
footage_socket.connect('tcp://localhost:5555')
#footage_socket.connect('tcp://192.168.0.25:5555')


context1 = zmq.Context()
footage_socket1 = context1.socket(zmq.SUB)
footage_socket1.bind('tcp://*:5556')
footage_socket1.setsockopt_string(zmq.SUBSCRIBE, np.unicode(''))



def main(yolo):

   # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0
    
   # deep_sort 
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    writeVideo_flag = True 
    
    #video_capture = cv2.VideoCapture(0)

    if writeVideo_flag:
    # Define the codec and create VideoWriter object
        #w = int(video_capture.get(3))
        #h = int(video_capture.get(4))
        w = 640
        h = 480
        fourcc = cv2.cv.CV_FOURCC(*'MJPG')
        #fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('output.avi', fourcc, 15, (w, h))
        list_file = open('detection.txt', 'w')
        frame_index = -1 
        
    fps = 0.0
    while True:
        #ret, frame = video_capture.read()  # frame shape 640*480*3
        #if ret != True:
         #   break;
        t1 = time.time()
        
        try:
        	frame1 = footage_socket1.recv_string()
        	img = base64.b64decode(frame1)
        	npimg = np.fromstring(img, dtype=np.uint8)
        	frame = cv2.imdecode(npimg, 1)
        	#cv2.imshow("Stream", source)
        	#cv2.waitKey(1)
        	image = Image.fromarray(frame)
        	boxs = yolo.detect_image(image)
		   # print("box_num",len(boxs))
        	features = encoder(frame,boxs)
		    
		    # score to 1.0 here).
        	detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
		    
		    # Run non-maxima suppression.
        	boxes = np.array([d.tlwh for d in detections])
        	scores = np.array([d.confidence for d in detections])
        	indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        	detections = [detections[i] for i in indices]
		    
		    # Call the tracker
        	tracker.predict()
        	tracker.update(detections)
		    
        	for track in tracker.tracks:
		        if track.is_confirmed() and track.time_since_update >1 :
		            continue 
		        bbox = track.to_tlbr()
		        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
		        cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)

        	for det in detections:
		        bbox = det.to_tlbr()
		        cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)
		        
        	cv2.imshow('', frame)
		    
        	if writeVideo_flag:
		        # save a frame
		        out.write(frame)
		        frame_index = frame_index + 1
		        list_file.write(str(frame_index)+' ')
		        if len(boxs) != 0:
		            for i in range(0,len(boxs)):
		                list_file.write(str(boxs[i][0]) + ' '+str(boxs[i][1]) + ' '+str(boxs[i][2]) + ' '+str(boxs[i][3]) + ' ')
		        list_file.write('\n')
		        
		        encoded, buffer = cv2.imencode('.jpg', frame)
		        jpg_as_text = base64.b64encode(buffer)
		        footage_socket.send(jpg_as_text)
		        
        	fps  = ( fps + (1./(time.time()-t1)) ) / 2
        	print("fps= %f"%(fps))
		    
		    # Press Q to stop!
        	if cv2.waitKey(1) & 0xFF == ord('q'):
		        break
        except KeyboardInterrupt:
       	 	cv2.destroyAllWindows()
       	 	break
        	

    video_capture.release()
    #if writeVideo_flag:
       # out.release()
      #  list_file.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(YOLO())
