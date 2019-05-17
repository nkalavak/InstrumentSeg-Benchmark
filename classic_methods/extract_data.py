import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.segmentation import *
import prepData
#from classicMethods import nonNNAlgo
from sklearn.decomposition import PCA
import pickle
import glob

# benchmark_data folder arrangement: 6 folders named as Surgery0, Surgery1, Surgery2, Surgery3, 
# Surgery4, Surgery5; each folder contains two folders, train and test, for training
# and test, respectively; in train/test folder, original image and the corresponding 
# mask have the same filename, while the original image is .jpg and the mask is .png.

def exData(save_idx = True):
    
#    train_x_all = []
#    train_y_all = []
#    train_sup_x_all = []
#    train_sup_y_all = []
#    seg_list_all = []
#    train_x_sub_all = []
#    train_y_sub_all = []
#    
#    test_x_all = []
#    test_y_all = []
#    test_sup_x_all = []
#    test_sup_y_all = []
#    test_seg_list_all = []
    
    for k in range(3,6):
        
        train_path = glob.glob('./benchmark_data/Surgery'+str(k)+'/train/*.jpg')
        test_path = glob.glob('./benchmark_data/Surgery'+str(k)+'/test/*.jpg')
        
#        train_x, train_y = prepData.getDataset(train_path)
#        train_sup_x, train_sup_y, seg_list, train_x_sub, train_y_sub = prepData.getSuperpixelDataset(train_path,train_x, train_y)
        num_frame = 100
        idx_list = np.arange(0, len(train_path), num_frame)
        for i in range(0,len(idx_list)):
            train_x, train_y = prepData.getDataset(train_path[idx_list[i]:(idx_list[i]+num_frame)])
            train_sup_x, train_sup_y, seg_list, train_x_sub, train_y_sub = \
            prepData.getSuperpixelDataset(train_path[idx_list[i]:(idx_list[i]+num_frame)],train_x, train_y) 
            
#            train_x_all.append(train_x)
#            train_y_all.append(train_y)
#            train_sup_x_all.append(train_sup_x)
#            train_sup_y_all.append(train_sup_y)
#            seg_list_all.append(seg_list)
#            train_x_sub_all.append(train_x_sub)
#            train_y_sub_all.append(train_y_sub)
            
            if save_idx is True:
                with open('train_data_surgery'+str(k)+'_'+str(i)+'.pkl', 'wb') as f:
                    pickle.dump( [train_x,train_y], f )
                    
                with open('train_data_short_surgery'+str(k)+'_'+str(i)+'.pkl', 'wb') as f:
                    pickle.dump( [train_x_sub, train_y_sub, train_sup_x,train_sup_y,seg_list], f )
                    
        test_x, test_y = prepData.getDataset(test_path)
        test_sup_x, test_sup_y, test_seg_list, _, _ = prepData.getSuperpixelDataset(test_path,test_x, test_y)
        
#        test_x_all.append(test_x)
#        test_y_all.append(test_y)
#        test_sup_x_all.append(test_sup_x)
#        test_sup_y_all.append(test_sup_y)
#        test_seg_list_all.append(test_seg_list)
        
        if save_idx is True:
            with open('test_data_surgery'+str(k)+'.pkl', 'wb') as f:
                pickle.dump( [test_x,test_y,test_sup_x,test_sup_y,test_seg_list], f )
        
#    return train_x_all, train_y_all, train_sup_x_all, train_sup_y_all, \
#    seg_list_all, train_x_sub_all, train_y_sub_all, test_x_all, test_y_all, \
#    test_sup_x_all, test_sup_y_all, test_seg_list_all

exData()