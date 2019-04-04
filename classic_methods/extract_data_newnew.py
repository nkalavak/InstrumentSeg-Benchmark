import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.segmentation import *
import prepData
from classicMethods import nonNNAlgo
from sklearn.decomposition import PCA
import pickle

#k = 5
#file_range_list = [(0,75),(75,150),(150,225)]

# for video 0
#file_range_list = [(0,100),(100,200),(200,300),(300,400),(400,500),(500,581)]

for k in range(0,6):
    if k == 0:
        file_range_list = [(0,100),(100,200),(200,300),(300,400),(400,500),(500,581)]
    else:
        file_range_list = [(0,75),(75,150),(150,225)]
    for i in range (0,len(file_range_list)):
        train_x, train_y = prepData.getDataset('./benchmark_data/s'+str(k)+'/train/*.jpg',file_range=file_range_list[i])
        train_sup_x, train_sup_y, _, train_x_sub, train_y_sub = prepData.getSuperpixelDataset('./benchmark_data/s'+str(k)+'/train/*.jpg',train_x, train_y,file_range=file_range_list[i])
        with open('s'+str(k)+'train'+str(i)+'_short.pkl', 'wb') as f:
            pickle.dump( [train_sup_x, train_sup_y, train_x_sub, train_y_sub], f )
    
    test_x, test_y = prepData.getDataset('./benchmark_data/s'+str(k)+'/test/*.jpg',train = False)
    test_sup_x, test_sup_y, test_segments_list, _, _ = prepData.getSuperpixelDataset('./benchmark_data/s'+str(k)+'/test/*.jpg',test_x, test_y)
    
    with open('s'+str(k)+'test.pkl', 'wb') as f:
        pickle.dump( [test_x, test_y, test_sup_x, test_sup_y, test_segments_list], f )
