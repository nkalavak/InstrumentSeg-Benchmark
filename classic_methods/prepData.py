import numpy as np
import cv2
#from sklearn.preprocessing import *
from skimage import feature
from skimage.filters import gabor_kernel
from scipy import ndimage as ndi
from timeit import default_timer as timer
import glob
from skimage.segmentation import *
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import random

def exFeature(name,height,width,offset):
    
    # extract label
    mask = cv2.imread(name[:-4]+'.png') # read the mask
    mask = cv2.cvtColor( mask, cv2.COLOR_BGR2GRAY )
    _,mask = cv2.threshold(mask,100,1,cv2.THRESH_BINARY) # extract label map
    mask = mask[offset:height-offset-1, offset:width-offset-1]
    label = np.reshape(mask,(-1,1),'F')
    
    # extract feature
    img = cv2.imread(name) 
    img_gray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
    
    # color feature O2, O3
    img_temp = img[offset:height-offset-1, offset:width-offset-1]
    f0_temp = np.reshape(img_temp,(-1,3),'F')
    f0 = np.zeros((f0_temp.shape[0],2))
    f0[:,0] = f0_temp[:,1] - f0_temp[:,0]
    f0[:,1] = f0_temp[:,2] - (f0_temp[:,1] + f0_temp[:,0])

    # color feature HSV
    img_hsv = cv2.cvtColor( img, cv2.COLOR_BGR2HSV )
    img_hsv = img_hsv[offset:height-offset-1, offset:width-offset-1]
    f1 = np.reshape(img_hsv,(-1,3),'F')
    
    # hog
    img_hog = myhog(img_gray,height,width,offset)
    f2 = np.reshape(img_hog,(-1,9),'F')
    
#    # local binary pattern
#    METHOD = 'uniform'
#    radius = 3
#    n_points = 8 * radius
#    img_lbp = lbp(img_gray,height,width,offset)
#    f3 = np.reshape(img_lbp,(-1,15),'F')

    return np.concatenate([f0,f1,f2,label],axis=1)

def exLabel(name,height,width,offset):
    mask = cv2.imread(name[:-4]+'.png') # read the mask
    mask = cv2.cvtColor( mask, cv2.COLOR_BGR2GRAY )
    _,mask = cv2.threshold(mask,100,1,cv2.THRESH_BINARY) # extract label map
    mask = mask[offset:height-offset-1, offset:width-offset-1]
    label = np.reshape(mask,(-1,1),'F')
    
    return label

def lbp(img,height,width,offset):
    
    METHOD = 'ror'
    radius = 3
    n_points = 8 * radius
    img_lbp = feature.local_binary_pattern(img, n_points, radius, METHOD)
    #new_img = img_lbp[offset:height-offset-1, offset:width-offset-1]
    
#    new_img_temp = Parallel(n_jobs=-1)(delayed(lbphist)(img_lbp,offset,i,j) \
#                            for i in range(offset,height-offset-1) for j in range(offset,width-offset-1))
    new_img_temp = [lbphist(img_lbp,offset,i,j) \
                            for i in range(offset,height-offset-1) for j in range(offset,width-offset-1)]
    new_img = np.reshape(new_img_temp,(height - 2*offset -1,width - 2*offset -1,15))

    #new_img = Parallel(n_jobs=-1)(delayed(lbpHist)(img_lbp,i,j,height,width,offset) \
#for j in range(0,width)for i in range(0,height))
    #new_img = np.concatenate( new_img, axis=0 )

    #new_img = np.zeros((height-2*offset-1,width-2*offset-1))
    
    #for i in range (0,height-2*offset-1):
    #    for j in range (0,width-2*offset-1):
    #        new_img[i,j] = np.mean(img_lbp[min(i-offset,0): max(i+offset,img_lbp.shape[0]),\
    #                                       min(j-offset,0): max(j+offset,img_lbp.shape[1])])
    
    return new_img

def lbphist(img_lbp,offset,i,j):
    hist_temp,_ = np.histogram(img_lbp[min(i-offset,0): max(i+offset,img_lbp.shape[0]),\
                                       min(j-offset,0): max(j+offset,img_lbp.shape[1])].ravel(), bins=15)
    return hist_temp
    
def lbpHist(img_lbp,i,j,height,width,offset):
    
    return np.mean(img_lbp[min(i-offset,0): max(i+offset,img_lbp.shape[0]),\
                                           min(j-offset,0): max(j+offset,img_lbp.shape[1])])
def myhog(img,height,width,offset):
    
    new_img = np.zeros((height-2*offset-1,width-2*offset-1,9))
    
    for i in range (0,8):
        for j in range (0,8):
            img_temp = img[i:,j:]
            hog_temp = np.squeeze(feature.hog(img_temp, cells_per_block=(1, 1), feature_vector=False))
            new_img[i::8,j::8,:] += hog_temp
    
    return new_img

def power(image, kernel):
    # Normalize images for better comparison.
    image = (image - image.mean()) / image.std()
    return np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap')**2 +
                   ndi.convolve(image, np.imag(kernel), mode='wrap')**2)
def Gabor(img):
    start = timer()
    num_direction = 4
    height, width  = img.shape
    new_img = np.zeros([height, width, num_direction])
    frequency = 0.1
    for i in range (0, num_direction):
        theta = i / num_direction * np.pi
        kernel = gabor_kernel(frequency, theta=theta)
        new_img[:,:,i] = power(img, kernel)
    print(timer()-start)
                
# read all images and extract features
# image '.jpg'; label: name+'.png'
# features: RGB
def getDataset(file_list):

    start = timer()
    
    width = 520
    height = 475
    offset = 3
    
#    all_data_list = [exFeature(file,height,width,offset) for file in file_list]
    all_data_list = Parallel(n_jobs=-1)(delayed(exFeature)(file,height,width,offset) for file in file_list) # parallel implementation
    all_data_list = np.concatenate( all_data_list, axis=0 )
    data_list = all_data_list[:,:-1]
    label_list = all_data_list[:,-1]
        
    print('get data', timer()-start)
    
    return data_list, label_list

# segment all images into superpixels (SLIC)
# extract the features of the superpixels (mean, variance of RGB)
# maybe in the future: fusion with getDataset()
def getSuperpixelDataset(file_list, featVec, pixel_label_list):

    start = timer()

    pixel_label_list = pixel_label_list.astype('int64')

    width = 520
    height = 475
    offset = 3
    num_pixel = (width-7)*(height-7)
    
#    seg_list = [getSuperpixel(file, height, width, offset) for file in file_list]
    seg_list = Parallel(n_jobs=-1)(delayed(getSuperpixel)(file, height, width, offset) for file in file_list) # parallel implementation
    # seg_list: each row corresponding to 1 image; each element is the superpixel index
    seg_list = np.concatenate( seg_list, axis=0 ) 
    
    data_list_all = Parallel(n_jobs=-1)(delayed(exSuperpixel)(file_list[i], featVec[i*num_pixel:(i+1)*num_pixel], seg_list[i], pixel_label_list[i*num_pixel:(i+1)*num_pixel]) \
                                    for i in range(0,len(seg_list))) 
    data_list_all = np.concatenate( data_list_all, axis=0 )
    
    data_list_temp1 = data_list_all[0::2]
    data_list = data_list_temp1[:,:-1]
    label_list = data_list_temp1[:,-1]
    
    data_list_temp2 = data_list_all[1::2]
    data_list2 = data_list_temp2[:,0:featVec.shape[1]]
    label_list2 = data_list_temp2[:,-1]
    
    print('get sup data', timer()-start)

    return data_list, label_list, seg_list, data_list2, label_list2

def getSuperpixel(name, height, width, offset):
    
    frame = cv2.imread(name)
    
    # cut the edge
    frame = frame[offset:height-offset-1, offset:width-offset-1]
    
    # SLIC
    # num of resultant segments will be less than n_segments, different for each image
    segments_slic = slic(frame, n_segments=200, compactness=10, sigma=1)
    segments_slic = np.reshape(segments_slic,(1,-1),'F')
    
    return segments_slic #np.concatenate([segments_slic,label])

def exSuperpixel(name, featVec, seg, plabel):
    
    N = len(np.unique(seg))
    new_featVec = Parallel(n_jobs=-1)(delayed(exSPfeature)(i,featVec,seg,plabel) for i in range (0,N))
    new_featVec = np.concatenate( new_featVec, axis=0 )
    
#     new_featVec = np.zeros((N,3))
#     for i in range (0,N):
#         idx = np.argwhere(seg == i)
#         new_featVec[i,0:2] = np.array( np.mean(featVec[idx]), np.var(featVec[idx]) )
        
#         plabel_temp = plabel[idx]
#         counts = np.bincount(plabel_temp.ravel())
#         new_featVec[i,2] = np.argmax(counts)
        
    return new_featVec

def exSPfeature(i,featVec,seg,plabel):
    
    num_feature = featVec.shape[1]
    
    # row0: features for superpixel; row1: features for other methods
    new_featVec = np.zeros([2,num_feature*2+1]) 
    idx = np.argwhere(seg == i)

    # extract mean and variance of each feature
    for k in range (0,featVec.shape[1]):
        new_featVec[0,k*2:k*2+2] = np.array( np.mean(featVec[idx,k]), np.var(featVec[idx,k]) )
    # label: majority vote    
    plabel_temp = plabel[idx]
    counts = np.bincount(plabel_temp.ravel())
    new_featVec[0,-1] = np.argmax(counts)
    
    # randomly pick a pixel within the superpixel 
    # for training classic algorithms other than superpixel 
    pick_idx = random.choice(idx)
    new_featVec[1,0:num_feature] = featVec[pick_idx,:]
    new_featVec[1,-1] = plabel[pick_idx]
    
    return new_featVec
