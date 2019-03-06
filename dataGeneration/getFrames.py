# extract frames from videos
import numpy as np
import cv2

cap = cv2.VideoCapture('Surgery2.mpg')
frame_rate = cap.get(5)
# num_frame = int( cap.get(7) )
# width = int( cap.get(3) )
# height = int( cap.get(4) )
# print "frame rate:", frame_rate # Frame Rate (fps)
# print "num of frames:", num_frame

count = 0
sampling_rate = 2
while(cap.isOpened()):
    frameId = cap.get(1) # current frame number
    _, frame = cap.read()
    
    if int( frameId % ( (1/sampling_rate) * frame_rate ) ) == 0:
        cv2.imwrite("frame%d.jpg"% count, frame)
        count += 1

cap.release()