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
K = 7 # fold
pca_list = [0,2,3,4,5]
gmmp_list = [[4,4],[3,4],[5,4],[4,3],[4,5]]
rf_list = [[20,10],[20,20],[20,30],[50,10],[100,10]]    
resultAna = nonNNAlgo(None,None,None,None,None,None,None)

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
    
non_NN_Algo = nonNNAlgo(None, None, None, None, None, None, framesize)
fold_size = int(train_x.shape[0]/K)
idx = list(range(0,train_x.shape[0], fold_size))

# train bayes-cv
print('start rf-cv...')

iou = []
dice = []
for rf_para in rf_list:
    non_NN_Algo = nonNNAlgo(None, None, None, None, None, rf_para, framesize)
    metrics = []
    for i in idx[:-1]:
        non_NN_Algo.train_X = train_x[i:(i+fold_size)]
        non_NN_Algo.train_Y = train_y[i:(i+fold_size)].ravel()
        
        if i == 0:
            non_NN_Algo.test_X = train_x[fold_size:]
            non_NN_Algo.test_Y = train_y[fold_size:].ravel()
        else:
            non_NN_Algo.test_X = np.concatenate([train_x[0:i], train_x[(i+fold_size):]],axis=0)
            non_NN_Algo.test_Y = np.concatenate([train_y[0:i], train_y[(i+fold_size):]],axis=0).ravel()
        
        pred_y = non_NN_Algo.randomForestSeg(train_idx = True, test_idx = True)
        metrics.append(resultAna.metricLoop(pred_y,non_NN_Algo.test_Y))
    
    metrics = np.concatenate(metrics,axis=0)    
    print(rf_para, np.mean(metrics[:,0]), np.mean(metrics[:,1]) )

## train bayes-cv
#print('start bayes-cv...')
#
#iou = []
#dice = []
#for pca_comp in pca_list:
#    metrics = []
#    for i in idx[:-1]:
#        # pca
#        if pca_comp > 0:
#            pca = PCA(n_components=pca_comp)
#            train_x_temp = pca.fit_transform(train_x[i:(i+fold_size)])
#        else:
#            train_x_temp = train_x[i:(i+fold_size)]
#        
#        non_NN_Algo.train_X = train_x_temp
#        non_NN_Algo.train_Y = train_y[i:(i+fold_size)].ravel()
#        
#        if i == 0:
#            if pca_comp > 0:
#                non_NN_Algo.test_X = pca.transform(train_x[fold_size:])
#            else:
#                non_NN_Algo.test_X = train_x[fold_size:]
#            non_NN_Algo.test_Y = train_y[fold_size:].ravel()
#        else:
#            if pca_comp > 0:
#                non_NN_Algo.test_X = pca.transform(np.concatenate([train_x[0:i], train_x[(i+fold_size):]],axis=0))
#            else:
#                non_NN_Algo.test_X = np.concatenate([train_x[0:i], train_x[(i+fold_size):]],axis=0)
#            non_NN_Algo.test_Y = np.concatenate([train_y[0:i], train_y[(i+fold_size):]],axis=0).ravel()
#        
#        pred_y = non_NN_Algo.naiveBayesianSeg(train_idx = True, test_idx = True)
#        metrics.append(resultAna.metricLoop(pred_y,non_NN_Algo.test_Y))
#    
#    metrics = np.concatenate(metrics,axis=0)    
#    print(pca_comp, np.mean(metrics[:,0]), np.mean(metrics[:,1]) )
#
#print('start gmm cv...')
#
#for gmm_para in gmmp_list:
#    non_NN_Algo = nonNNAlgo(None, None, None, None, gmm_para, None, framesize)    
#    for pca_comp in pca_list:
#        metrics = []
#        for i in idx[:-1]:
#            # pca
#            if pca_comp > 0:
#                pca = PCA(n_components=pca_comp)
#                train_x_temp = pca.fit_transform(train_x[i:(i+fold_size)])
#            else:
#                train_x_temp = train_x[i:(i+fold_size)]
#            
#            non_NN_Algo.train_X = train_x_temp
#            non_NN_Algo.train_Y = train_y[i:(i+fold_size)].ravel()
#            
#            if i == 0:
#                if pca_comp > 0:
#                    non_NN_Algo.test_X = pca.transform(train_x[fold_size:])
#                else:
#                    non_NN_Algo.test_X = train_x[fold_size:]
#                non_NN_Algo.test_Y = train_y[fold_size:].ravel()
#            else:
#                if pca_comp > 0:
#                    non_NN_Algo.test_X = pca.transform(np.concatenate([train_x[0:i], train_x[(i+fold_size):]],axis=0))
#                else:
#                    non_NN_Algo.test_X = np.concatenate([train_x[0:i], train_x[(i+fold_size):]],axis=0)
#                non_NN_Algo.test_Y = np.concatenate([train_y[0:i], train_y[(i+fold_size):]],axis=0).ravel()
#            
#            pred_y = non_NN_Algo.GaussianMixtureModelSeg(train_idx = True, test_idx = True)
#            metrics.append(resultAna.metricLoop(pred_y,non_NN_Algo.test_Y))
#        
#        metrics = np.concatenate(metrics,axis=0)    
#        print(gmm_para, pca_comp, np.mean(metrics[:,0]), np.mean(metrics[:,1]) )

## superpixel  
#train_x = [] # init train_x
#train_y = [] # init train_y
#for k in range (0,6):# read every video
#    for filename in glob.glob('s'+str(k)+'train*_short2.pkl'):
#        with open(filename, 'rb') as f:
#            train_x_temp, train_y_temp, _,_ = pickle.load( f )
#        train_x.append(train_x_temp)
#        train_y.append(train_y_temp) 
#train_x = np.concatenate(train_x, axis=0)    
#train_y = np.concatenate(train_y, axis=0)
#
#fold_size = int(train_x.shape[0]/K)
#idx = list(range(0,train_x.shape[0], fold_size))
#        
#print('start superpixel cv...')
#for gmm_para in gmmp_list:
#    non_NN_Algo = nonNNAlgo(None, None, None, None, gmm_para, None, framesize)    
#    for pca_comp in pca_list:
#        metrics = []
#        for i in idx[:-1]:
#            # pca
#            if pca_comp > 0:
#                pca = PCA(n_components=pca_comp)
#                train_x_temp = pca.fit_transform(train_x[i:(i+fold_size)])
#            else:
#                train_x_temp = train_x[i:(i+fold_size)]
#            
#            non_NN_Algo.train_X = train_x_temp
#            non_NN_Algo.train_Y = train_y[i:(i+fold_size)].ravel()
#            
#            if i == 0:
#                if pca_comp > 0:
#                    non_NN_Algo.test_X = pca.transform(train_x[fold_size:])
#                else:
#                    non_NN_Algo.test_X = train_x[fold_size:]
#                non_NN_Algo.test_Y = train_y[fold_size:].ravel()
#            else:
#                if pca_comp > 0:
#                    non_NN_Algo.test_X = pca.transform(np.concatenate([train_x[0:i], train_x[(i+fold_size):]],axis=0))
#                else:
#                    non_NN_Algo.test_X = np.concatenate([train_x[0:i], train_x[(i+fold_size):]],axis=0)
#                non_NN_Algo.test_Y = np.concatenate([train_y[0:i], train_y[(i+fold_size):]],axis=0).ravel()
#            
#            pred_y = non_NN_Algo.GaussianMixtureModelSeg(train_idx = True, test_idx = True)
#            metrics.append(resultAna.metricLoop(pred_y,non_NN_Algo.test_Y))
#        
#        metrics = np.concatenate(metrics,axis=0)    
#        print(gmm_para, pca_comp, np.mean(metrics[:,0]), np.mean(metrics[:,1]) )