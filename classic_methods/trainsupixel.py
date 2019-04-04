import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.segmentation import *
import prepData
from classicMethods import nonNNAlgo
from sklearn.decomposition import PCA
import pickle
from timeit import default_timer as timer
import glob

framesize = (480,720)
width = 520
height = 475
num_pixel = (width-7)*(height-7)

# superpixel segmentation
# read data
train_x = [] # init train_x
train_y = [] # init train_y
for k in range (0,6):# read every video
    for filename in glob.glob('s'+str(k)+'train*_short2.pkl'):
        with open(filename, 'rb') as f:
            train_x_temp, train_y_temp, _,_ = pickle.load( f )
        train_x.append(train_x_temp)
        train_y.append(train_y_temp) 

#pca = PCA(n_components=2)
#train_x = pca.fit_transform(train_x)
       
train_x = np.concatenate(train_x, axis=0)    
train_y = np.concatenate(train_y, axis=0)
non_NN_Algo = nonNNAlgo(None, None, None, None, (4,3),(5,5),framesize)
non_NN_Algo.train_X = train_x
non_NN_Algo.train_Y = train_y.ravel()
print('start training superpixel...')
start = timer()
_ = non_NN_Algo.GaussianMixtureModelSeg(train_idx = True, test_idx = False)
print('[superpixel]', timer() - start)

print('test superpixel...')
img_num_list = [146,75,75,75,75,75]
iou = 0
dice = 0
test_time = 0
for k in range (0,6):
    with open('s'+str(k)+'test2.pkl', 'rb') as f:
        _, test_y, test_x, _, test_segments_list = pickle.load( f )

    start = timer()
    non_NN_Algo.test_X = test_x # pca.transform(test_x)
    non_NN_Algo.test_Y = test_y.ravel()
    sup_test_y_pred = non_NN_Algo.GaussianMixtureModelSeg(train_idx = False, test_idx = True)
    # calculate the metrics
    # print(test_segments_list.shape, pixel_sup_test_y.shape,test_sup_y.shape,sup_test_y_pred.shape,len(np.unique(test_segments_list)))
    sup_offset = 0
    resultAna = nonNNAlgo(None,None,None,None,(6,3),(10,5),None)
    metrics = []
    for j in range (0, len(test_segments_list) ):
        pixel_sup_test_y = np.zeros((num_pixel,1))
        seg_list_temp = test_segments_list[j]
        for i in np.unique(seg_list_temp):
            idx = np.argwhere(seg_list_temp == i)
            pixel_sup_test_y[idx] = sup_test_y_pred[sup_offset+i]
        metrics.append(resultAna.metricLoop(np.squeeze(pixel_sup_test_y), test_y[num_pixel*j:num_pixel*(j+1)]))
        sup_offset += len(np.unique(seg_list_temp))
    test_time += timer() - start

    metrics = np.concatenate(metrics, axis=0)
    #print('iou',np.mean(metrics[:,0]),'dice',np.mean(metrics[:,1]))
    iou += np.mean(metrics[:,0])*img_num_list[k]
    dice += np.mean(metrics[:,1])*img_num_list[k]

print('1)iou:',iou/521,'2)dice:',dice/521,'3)time:',test_time/521)
