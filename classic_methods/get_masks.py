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

# read all data
train_x = [] # init train_x
train_y = [] # init train_y
for k in range (0,6):# read every video
    for filename in glob.glob('s'+str(k)+'train*_short2.pkl'):
        with open(filename, 'rb') as f:
            _, _, train_x_temp, train_y_temp = pickle.load( f )
        train_x.append(train_x_temp)
        train_y.append(train_y_temp)        
train_x = np.concatenate(train_x, axis=0)    
train_y = np.concatenate(train_y, axis=0)
        
non_NN_Algo = nonNNAlgo(None, None, None, None, (4,3),(100,20),framesize)
non_NN_Algo.train_X = train_x
non_NN_Algo.train_Y = train_y.ravel()
# train rf
print('start training rf...')
start = timer() 
_ = non_NN_Algo.randomForestSeg(train_idx = True, test_idx = False)
print('[rf]', timer() - start)

with open('mask_test.pkl', 'rb') as f:
    test_x, test_y,_, _,_ = pickle.load( f )
non_NN_Algo.test_X = test_x
non_NN_Algo.test_Y = test_y.ravel()

start = timer()
test_results = non_NN_Algo.randomForestSeg(train_idx = False, test_idx = True)
rf_time = timer() - start
rf_pred = test_results

# pca for bayes
pca = PCA(n_components=5)
train_x_pca = pca.fit_transform(train_x)
non_NN_Algo.train_X = train_x_pca

# train Bayes
print('start training Bayes...')
start = timer()
_ = non_NN_Algo.naiveBayesianSeg(train_idx = True, test_idx = False)
print('[bayes]', timer() - start)

start = timer()
non_NN_Algo.test_X = pca.transform(test_x)
test_results = non_NN_Algo.naiveBayesianSeg(train_idx = False, test_idx = True)
bayes_time = timer() - start
bayes_pred = test_results

# pca for gmm
pca = PCA(n_components=5)
train_x_pca = pca.fit_transform(train_x)
non_NN_Algo.train_X = train_x_pca

print('start training gmm...')
_ = non_NN_Algo.GaussianMixtureModelSeg(train_idx = True, test_idx = False)

start = timer()
non_NN_Algo.test_X = pca.transform(test_x)
test_results = non_NN_Algo.GaussianMixtureModelSeg(train_idx = False, test_idx = True)
gmm_time = timer() - start
gmm_pred = test_results

path = './test_mask_dataset/*.jpg'
idx = list(range(0,num_pixel*7, num_pixel))
for filename,i in zip(glob.glob(path),idx):
    start = timer()
    pred_mask = 255*np.reshape(rf_pred[i:(i+num_pixel)],(height-7,width-7),'F')
    rf_time += (timer()-start)
    cv2.imwrite('./test_mask_results/rf_'+filename[20:],pred_mask)
    
    start = timer()
    pred_mask = 255*np.reshape(bayes_pred[i:(i+num_pixel)],(height-7,width-7),'F')
    bayes_time += (timer()-start)
    cv2.imwrite('./test_mask_results/bayes_'+filename[20:],pred_mask)
    
    start = timer()
    pred_mask = 255*np.reshape(gmm_pred[i:(i+num_pixel)],(height-7,width-7),'F')
    gmm_time += (timer()-start)
    cv2.imwrite('./test_mask_results/gmm_'+filename[20:],pred_mask)

print(rf_time/7)
print(bayes_time/7) 
print(gmm_time/7)

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
        
train_x = np.concatenate(train_x, axis=0)    
train_y = np.concatenate(train_y, axis=0)
non_NN_Algo = nonNNAlgo(None, None, None, None, (4,3),None,framesize)
non_NN_Algo.train_X = train_x
non_NN_Algo.train_Y = train_y.ravel()
print('start training superpixel...')
_ = non_NN_Algo.GaussianMixtureModelSeg(train_idx = True, test_idx = False)

with open('mask_test.pkl', 'rb') as f:
    _, test_y, test_x, _, test_segments_list = pickle.load( f )    
non_NN_Algo.test_X = test_x # pca.transform(test_x)
non_NN_Algo.test_Y = test_y.ravel()
sup_test_y_pred = non_NN_Algo.GaussianMixtureModelSeg(train_idx = False, test_idx = True)

sup_time = 0
sup_offset = 0
resultAna = nonNNAlgo(None,None,None,None,(2,2),(2,2),None)
filelist = glob.glob(path)
for j in range (0, len(test_segments_list) ):
    start = timer()
    pixel_sup_test_y = np.zeros((num_pixel,1))
    seg_list_temp = test_segments_list[j]
    for i in np.unique(seg_list_temp):
        idx = np.argwhere(seg_list_temp == i)
        pixel_sup_test_y[idx] = sup_test_y_pred[sup_offset+i]
    pred_mask = 255*np.reshape(pixel_sup_test_y,(height-7,width-7),'F')
    sup_time += (timer()-start)
    cv2.imwrite('./test_mask_results/superpixel_'+filelist[j][20:],pred_mask)
    sup_offset += len(np.unique(seg_list_temp))
print(sup_time/7)