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
#import extract_data

width = 520
height = 475
num_pixel = (width-7)*(height-7)

# extract train and test data directly
# train_x, train_y, _, _, _, _, _, test_x, test_y, _, _, _ = extract_data.exData(save_idx = False)

# read all data
train_x = []
train_y = []
test_x = []
test_y = []
for k in range (0,4):# read every video
    
    # train data
    for filename in glob.glob('train_data_surgery'+str(k)+'_*.pkl'):
        with open(filename, 'rb') as f:
            train_x_temp, train_y_temp = pickle.load( f )
#            train_x_temp, train_y_temp, _, _, _ = pickle.load( f ) # for _short
        train_x.append(train_x_temp)
        train_y.append(train_y_temp) 
        print(train_x_temp.shape,train_y_temp.shape)
    
    # test data
    with open('test_data_surgery'+str(k)+'.pkl', 'rb') as f:
        test_x_temp, test_y_temp, _, _,_ = pickle.load( f )
    test_x.append(test_x_temp)
    test_y.append(test_y_temp)
    
    print('test', test_x_temp.shape, test_x_temp.shape)
    
train_x = np.concatenate(train_x, axis=0)    
train_y = np.concatenate(train_y, axis=0)
test_x = np.concatenate(test_x, axis=0)    
test_y = np.concatenate(test_y, axis=0)
 
# print(train_x.shape,train_y.shape, test_x.shape, test_y.shape)
   
## pca: only works for _short data
#pca = PCA(n_components=5)
#train_x = pca.fit_transform(train_x)
        
non_NN_Algo = nonNNAlgo(None, None, None, None, None, None)
non_NN_Algo.train_X = train_x
non_NN_Algo.train_Y = train_y.ravel()

# train Bayes
print('start training Bayes...')
start = timer()
_ = non_NN_Algo.naiveBayesianSeg(train_idx = True, test_idx = False)
print('[bayes]', timer() - start)

## test
#non_NN_Algo.test_X = pca.transform(test_x)
non_NN_Algo.test_X = test_x
non_NN_Algo.test_Y = test_y.ravel()
test_results = non_NN_Algo.naiveBayesianSeg(train_idx = False, test_idx = True, pMetrics=True)
print('test bayes: 1)iou:',test_results[1],'2)dice:',test_results[2],'3)time:',test_results[3])