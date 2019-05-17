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
train_x = []
train_y = []
test_x = []
test_y = []
for k in range (0,3):# read every video
    
    # train data
    for filename in glob.glob('train_data_surgery'+str(k)+'_*.pkl'):
        with open(filename, 'rb') as f:
            train_x_temp, train_y_temp = pickle.load( f )
#            train_x_temp, train_y_temp, _, _, _ = pickle.load( f ) # for _short
        train_x.append(train_x_temp)
        train_y.append(train_y_temp) 
    
    # test data
    with open('test_data_surgery'+str(k)+'.pkl', 'rb') as f:
        test_x_temp, test_y_temp, _, _,_ = pickle.load( f )
    test_x.append(test_x_temp)
    test_y.append(test_y_temp)
    
train_x = np.concatenate(train_x, axis=0)    
train_y = np.concatenate(train_y, axis=0)
test_x = np.concatenate(test_x, axis=0)    
test_y = np.concatenate(test_y, axis=0)

# extract part of the data
train_x = train_x[0:91000000,:]
train_y = train_y[0:91000000]

print(train_x.shape,train_y.shape, test_x.shape, test_y.shape)
        
non_NN_Algo = nonNNAlgo(None, None, None, None, (4,3),(2,2),framesize)
non_NN_Algo.train_X = train_x
non_NN_Algo.train_Y = train_y.ravel()

## train gmm
# pca for gmm
pca = PCA(n_components=5)
train_x_pca = pca.fit_transform(train_x)
non_NN_Algo.train_X = train_x_pca
non_NN_Algo.train_Y = train_y.ravel()

print('start training gmm...')
start = timer() 
_ = non_NN_Algo.GaussianMixtureModelSeg(train_idx = True, test_idx = False)
print('[gmm]', timer() - start)

### test
#print('test gmm...')
#non_NN_Algo.test_X = pca.fit_transform(test_x)
#non_NN_Algo.test_Y = test_y.ravel()
#test_results = non_NN_Algo.GaussianMixtureModelSeg(train_idx = False, test_idx = True, pMetrics=True)
#print('test bayes: 1)iou:',test_results[1],'2)dice:',test_results[2],'3)time:',test_results[3])
    
## test gmm
#print('test gmm...')
#img_num_list = [146,75,75,75,75,75]
#gmm_iou =0
#gmm_dice =0
#gmm_time =0
#for k in range (0,6):
#    with open('s'+str(k)+'test2.pkl', 'rb') as f:
#        test_x, test_y,_, _,_ = pickle.load( f )
#    non_NN_Algo.test_X = pca.transform(test_x)
#    non_NN_Algo.test_Y = test_y.ravel()
#
#    test_results = non_NN_Algo.GaussianMixtureModelSeg(train_idx = False, test_idx = True, pMetrics=True)
#    gmm_iou += test_results[1]*img_num_list[k]
#    gmm_dice += test_results[2]*img_num_list[k]
#    gmm_time += test_results[3]
#
#print('test gmm: 1)iou:',gmm_iou/521,'2)dice:',gmm_dice/521,'3)time:',gmm_time/521)
