import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.segmentation import *
import prepData
from classicMethods import nonNNAlgo
from sklearn.decomposition import PCA
import pickle


test_x, test_y = prepData.getDataset('./test_mask_dataset/*.jpg',train = False)
test_sup_x, test_sup_y, test_segments_list, _, _ = prepData.getSuperpixelDataset('./test_mask_dataset/*.jpg',test_x, test_y)

with open('mask_test.pkl', 'wb') as f:
    pickle.dump( [test_x, test_y, test_sup_x, test_sup_y, test_segments_list], f )
