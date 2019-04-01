#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import division
import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import csv
import os, glob
# (0) define parameters
plt.ion()
first_entry = True
n_images = 700 # number of image frames in each video file
r_start = 0
c_start = 0
n_rows = 475 # including border: 1080
n_cols = 520 # including border: 1920
scale = 0.5
n_features = 2
n_color_components = 4
mask_max_value = 255
n_border_pixels = 70
edge_range = 5
blur_threshold1 = 150
blur_threshold2 = 450
include_top = True
include_bottom = True


# In[2]:


VERY_CLEAR = 1
A_LITTLE_BLUR = 2
VERY_BLURRY = 3

"""finddata_path = 'G:/Catalyst_March_Quals5/ColorStats_ToLoad'#'D:/01 Projects/AMAZON CATALYST PROJECT/Quals3/ColorStats_ToLoad'
savedata_path = 'G:/Catalyst_March_Quals5/Test/TotalStats_300instruments'#'D:/01 Projects/AMAZON CATALYST PROJECT/Quals4/Test/TotalStats_300instruments'
saveimag_path = 'G:/Catalyst_March_Quals5/Test/Test_results'#'D:/01 Projects/AMAZON CATALYST PROJECT/Quals4/Test/Test_results'"""

"""finddata_path = 'G:/Video_Debug/images_crop'
savedata_path = 'D:/01 Projects/AMAZON CATALYST PROJECT/Quals3/Test/TotalStats_300instruments'
saveimag_path = 'G:/Video_Debug/Test_results'"""


basedata_path = r'/home/nivii/Desktop/MICCAI2019_Dataset/Test/surgery'
image_location = 'Images/'
#savedata_path = r'/home/nivii/Desktop/Data/wicv/TotalStats_700instruments'
savedata_path = r'/home/nivii/Desktop/MICCAI2019_Dataset/Train/MergedStats'
basesaveimag_path = r'/home/nivii/Desktop/MICCAI2019_Dataset/Test/heuristic_predictions/surgery'
gt_location = 'ground_truth'

extension = r'*.jpg'

bg_prob_hs = np.load(savedata_path+'/bg_prob_hs.npy')       # shape:(n_hues, n_colors)
bg_prob_oppo = np.load(savedata_path+'/bg_prob_oppo.npy')   # shape:(n_colors, n_colors)

mask_max_value = 255
feature0_max_value = 100000
feature1_max_value = 10000
feature0_thres = 500
feature1_thres = 50
long_shape_ratio = 5 #1.5

# Start Segmentation
start_image_index = 0

#for im in range(start_image_index, n_images):
for i in range(1,7):
    finddata_path = basedata_path + str(i)
    saveimag_path = basesaveimag_path + str(i)
    print(finddata_path)
    print("Inside Folder Surgery: ", i)
    for pathAndFilename in glob.iglob(os.path.join(finddata_path, image_location, extension)):
        print("Reached here")
        # (1) Load image
        #im = 8
        #index = '%03d' % im
        #index = im
        #raw_image = cv2.imread(finddata_path+'/frame'+str(index)+'.jpg')
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))
        raw_image = cv2.imread(os.path.join(finddata_path, image_location, title + ext))
        image = np.array(raw_image)
        img = image[r_start:r_start+n_rows,c_start:c_start+ n_cols,:]
        img = img[...,[2,1,0]]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # compute the Laplacian of the image and then return the focus
        # measure, which is simply the variance of the Laplacian
        laplace_image = cv2.Laplacian(gray, cv2.CV_64F)
        blur_value = laplace_image.var()

        if blur_value < blur_threshold1:
            BLUR_STATUS = VERY_BLURRY
        elif blur_value < blur_threshold2:
            BLUR_STATUS = A_LITTLE_BLUR
        else:
            BLUR_STATUS = VERY_CLEAR

        # (2) Important color components
        image_oppo = np.zeros((np.shape(img)[0],np.shape(img)[1],2))    # shape: n_rows X n_cols X 2
        image_oppo[:,:,0] = (img[:,:,1]-img[:,:,0]+255)/2
        image_oppo[:,:,1] = (img[:,:,2]-(img[:,:,0]+img[:,:,1])+255*2)/3
        image_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)                   # shape: n_rows X n_cols X 3
        image_hs = image_hsv[:,:,0:2]

        # (3) prepare for grabcut
        img_small1 = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation = cv2.INTER_AREA)
        image_hsv_small = cv2.resize(image_hsv,(0,0), fx=scale, fy=scale, interpolation = cv2.INTER_AREA)
        mask1 = cv2.GC_PR_BGD*np.ones(img.shape[:2],np.uint8)
        bgdModel1 = np.zeros((1,65),np.float64)
        fgdModel1 = np.zeros((1,65),np.float64)
        rect1 = (0,0,0,0)

        # (4) prepare mask
        f = np.zeros((np.shape(img)[0],np.shape(img)[1],n_color_components), dtype=np.int8) # shape: n_rows X n_cols X n_color_components
        f[:,:,0:2] = image_hs
        f[:,:,2:4] = image_oppo

        P_label = np.zeros((np.shape(img)[0],np.shape(img)[1],n_features))
        P_label[:,:,0] = bg_prob_hs[f[:,:,0],f[:,:,1]]      # 2 dimensional lookup table
        P_label[:,:,1] = bg_prob_oppo[f[:,:,0],f[:,:,1]]    # 2 dimensional lookup table

        P_label[:,:,0] = feature0_max_value/np.max(P_label[:,:,0])*P_label[:,:,0]   # normalize to 0 ~ feature0_max_value
        P_label[:,:,1] = feature1_max_value/np.max(P_label[:,:,1])*P_label[:,:,1]   # normalize to 0 ~ feature1_max_value

        P_label = np.clip(P_label, 0, mask_max_value) #clips values outside of interval [0,mask_max_value]

        mask1 = np.where(np.logical_and(P_label[:,:,1] == feature1_thres,P_label[:,:,0] > feature0_thres), cv2.GC_FGD, mask1)
        mask1 = np.where(P_label[:,:,1] > feature1_thres, cv2.GC_BGD, mask1)

        # (5) compute edges
        tmp = cv2.medianBlur(np.uint8(P_label[:,:,1]), 31)
        sigma = 0.5
        v = np.median(P_label[:,:,1])
        edges = cv2.Canny(np.uint8(tmp),int(max(0,(1.0-sigma)*v)),int(min(255,(1.0+sigma)*v)),apertureSize = 3)
        kernel_dilate = np.ones((10,10), np.uint8)
        edges_dilate = cv2.dilate(edges, kernel_dilate, iterations=1)

        kernel_morph = np.ones((20,20), np.uint8)
        morph_open = cv2.morphologyEx(edges_dilate, cv2.MORPH_OPEN, kernel_morph)
        morph_clos = cv2.morphologyEx(edges_dilate, cv2.MORPH_CLOSE, kernel_morph)
        morph_open = 255-morph_open
        morph_clos = 255-morph_clos

        im2, contours, hierarchy = cv2.findContours(np.copy(morph_clos),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        hierarchy = hierarchy[0]

        edges_inner = np.zeros(img.shape[:2], np.uint8)
        edges_outer = np.zeros(img.shape[:2], np.uint8)

        for component in zip(contours, hierarchy):
            currentContour = component[0]
            currentHierarchy = component[1]
            if currentHierarchy[2] < 0: # no child
                cv2.drawContours(edges_inner, currentContour, -1, (1,1,1),-1)
            if currentHierarchy[3] < 0: # no parent
                cv2.drawContours(edges_outer, currentContour, -1, (1,1,1),-1)
        edges_outerinner = np.clip(edges_outer+edges_inner, 0, 1)
        edges_outerinner = cv2.resize(edges_outerinner, None, fx=scale, fy=scale, interpolation = cv2.INTER_NEAREST)

        # (6) grabcut with mask
        mask1 = cv2.resize(mask1, (0,0), fx=scale, fy=scale, interpolation = cv2.INTER_AREA)
        cv2.grabCut(img_small1, mask1, rect1, bgdModel1, fgdModel1, 5, cv2.GC_INIT_WITH_MASK)
        mask2 = np.where((mask1==cv2.GC_PR_BGD)|(mask1==cv2.GC_BGD),0,1).astype('uint8')
        im3, contours, hierarchy = cv2.findContours(np.copy(mask2),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        mask3 = np.zeros(img_small1.shape[:2], np.uint8)

        top_bound = 3
        left_bound = 3
        down_bound = img_small1.shape[0]-3
        right_bound = img_small1.shape[1]-3

        sat_border_values = np.concatenate((image_hsv_small[0,:,1], image_hsv_small[img_small1.shape[0]-1,:,1],                                image_hsv_small[:,0,1].T,image_hsv_small[:,img_small1.shape[1]-1,1].T), axis=0)
        sat_border_values = np.sort(sat_border_values,axis=None)
        sat_thres_index = np.sum(mask2[0,:])+np.sum(mask2[mask2.shape[0]-1,:])+np.sum(mask2[:,0])+np.sum(mask2[:,mask2.shape[1]-1])
        sat_thres = np.max([sat_border_values[np.min([int(sat_thres_index),len(sat_border_values)-1])],0.9*np.median(sat_border_values)])

        number_of_contours_drawn = 0
        relax_constraint = False
        iteration = 0

        while number_of_contours_drawn == 0 and iteration < 2:
            for ct in range(len(contours)):
                top_count = 0
                top_satavg = 255
                left_count = 0
                left_satavg = 255
                down_count = 0
                down_satavg = 255
                right_count = 0
                right_satavg = 255
                top_match_edge = False
                left_match_edge = False
                down_match_edge = False
                right_match_edge = False
                top_match_disp = False
                left_match_disp = False
                down_match_disp = False
                right_match_disp = False
                # conpute the outer bound of contour
                top_pts   = contours[ct][:,:,1]< top_bound
                left_pts  = contours[ct][:,:,0] < left_bound
                down_pts  = contours[ct][:,:,1] > down_bound
                right_pts = contours[ct][:,:,0] > right_bound

                top_cnt   = contours[ct][top_pts,0]
                left_cnt  = contours[ct][left_pts,1]
                down_cnt  = contours[ct][down_pts,0]
                right_cnt = contours[ct][right_pts,1]

                if  top_cnt.size > 0:
                    tmp_max = np.max(top_cnt)
                    tmp_min = np.min(top_cnt)
                    bder_range = np.arange(top_bound-3,top_bound+3+1)
                    non_bder_range1 = np.arange(np.max([tmp_max-edge_range,0]),np.min([tmp_max+edge_range,edges_outerinner.shape[1]]))
                    non_bder_range2 = np.arange(np.max([tmp_min-edge_range,0]),np.min([tmp_min+edge_range,edges_outerinner.shape[1]]))
                    index1 = np.ix_(bder_range,non_bder_range1)
                    index2 = np.ix_(bder_range,non_bder_range2)

                    top_match_edge = (edges_outerinner[index1].sum() + edges_outerinner[index2].sum()) > 0
                    #top_match_disp = (laplace_disparity_thres[index1].sum() > 0 or np.max(non_bder_range1)<tmp_max+edge_range) \
                    #               and (laplace_disparity_thres[index2].sum() > 0 or np.min(non_bder_range2)>tmp_min-edge_range)

                    top_count = tmp_max-tmp_min+1
                    top_satavg = np.median(image_hsv_small[contours[ct][top_pts,1],contours[ct][top_pts,0],1])

                if  left_cnt.size > 0:
                    tmp_max = np.max(left_cnt)
                    tmp_min = np.min(left_cnt)
                    bder_range = np.arange(left_bound-3,left_bound+3+1)
                    non_bder_range1 = np.arange(np.max([tmp_max-edge_range,0]),np.min([tmp_max+edge_range,edges_outerinner.shape[0]]))
                    non_bder_range2 = np.arange(np.max([tmp_min-edge_range,0]),np.min([tmp_min+edge_range,edges_outerinner.shape[0]]))
                    index1 = np.ix_(non_bder_range1,bder_range)
                    index2 = np.ix_(non_bder_range2,bder_range)

                    left_match_edge = (edges_outerinner[index1].sum() + edges_outerinner[index2].sum()) > 0
                    #left_match_disp = (laplace_disparity_thres[index1].sum() > 0 or np.max(non_bder_range1)<tmp_max+edge_range) \
                    #               and (laplace_disparity_thres[index2].sum() > 0 or np.min(non_bder_range2)>tmp_min-edge_range)

                    left_count = tmp_max-tmp_min+1
                    left_satavg = np.median(image_hsv_small[contours[ct][left_pts,1],contours[ct][left_pts,0],1])

                if  down_cnt.size > 0:
                    tmp_max = np.max(down_cnt)
                    tmp_min = np.min(down_cnt)
                    bder_range = np.arange(down_bound-3,down_bound+3)

                    non_bder_range1 = np.arange(np.max([tmp_max-edge_range,0]),np.min([tmp_max+edge_range,edges_outerinner.shape[1]]))
                    non_bder_range2 = np.arange(np.max([tmp_min-edge_range,0]),np.min([tmp_min+edge_range,edges_outerinner.shape[1]]))
                    index1 = np.ix_(bder_range,non_bder_range1)
                    index2 = np.ix_(bder_range,non_bder_range2)

                    down_match_edge = (edges_outerinner[index1].sum() + edges_outerinner[index2].sum())>0
                    #down_match_disp = (laplace_disparity_thres[index1].sum() > 0 or np.max(non_bder_range1)<tmp_max+edge_range) \
                    #               and (laplace_disparity_thres[index2].sum() > 0 or np.min(non_bder_range2)>tmp_min-edge_range)

                    down_count = tmp_max-tmp_min+1
                    down_satavg = np.median(image_hsv_small[contours[ct][down_pts,1],contours[ct][down_pts,0],1])

                if  right_cnt.size > 0:
                    tmp_max = np.max(right_cnt)
                    tmp_min = np.min(right_cnt)
                    bder_range = np.arange(right_bound-3,right_bound+3)
                    non_bder_range1 = np.arange(np.max([tmp_max-edge_range,0]),np.min([tmp_max+edge_range,edges_outerinner.shape[0]]))
                    non_bder_range2 = np.arange(np.max([tmp_min-edge_range,0]),np.min([tmp_min+edge_range,edges_outerinner.shape[0]]))
                    index1 = np.ix_(non_bder_range1,bder_range)
                    index2 = np.ix_(non_bder_range2,bder_range)

                    right_match_edge = (edges_outerinner[index1].sum() + edges_outerinner[index2].sum()) > 0
                    #right_match_disp = (laplace_disparity_thres[index1].sum() > 0 or np.max(non_bder_range1)<tmp_max+edge_range) \
                    #               and (laplace_disparity_thres[index2].sum() > 0 or np.min(non_bder_range2)>tmp_min-edge_range)

                    right_count = tmp_max-tmp_min+1
                    right_satavg = np.median(image_hsv_small[contours[ct][right_pts,1],contours[ct][right_pts,0],1])

                top_chk   = top_count > n_border_pixels
                left_chk  = left_count > n_border_pixels
                down_chk  = down_count > n_border_pixels
                right_chk = right_count > n_border_pixels

                if include_top:
                    border_chk = top_chk or down_chk or right_chk or left_chk
                    sat_chk  = top_satavg < sat_thres or left_satavg < sat_thres or down_satavg < sat_thres or right_satavg < sat_thres
                    edge_chk = top_match_edge or left_match_edge or down_match_edge or right_match_edge
                    #disp_chk = top_match_disp or left_match_disp or down_match_disp or right_match_disp

                else:
                    if include_bottom:
                        border_chk = down_chk or right_chk or left_chk
                        sat_chk  = left_satavg < sat_thres or down_satavg < sat_thres or right_satavg < sat_thres
                        edge_chk =  left_match_edge or down_match_edge or right_match_edge
                     #   disp_chk =  left_match_disp or down_match_disp or right_match_disp
                    else:
                        border_chk = right_chk or left_chk
                        sat_chk  = left_satavg < sat_thres  or right_satavg < sat_thres
                        edge_chk = left_match_edge or right_match_edge
                      #  disp_chk = left_match_disp or right_match_disp

                # shape constraint
                rect = cv2.minAreaRect(contours[ct])
                long_shape = max(rect[1])/max(min(rect[1]),1.0)>5.0
                

                if relax_constraint: # need to loosen constraint
                    if border_chk:
                        cv2.drawContours(mask3, contours, ct, 1,-1)
                        number_of_contours_drawn = number_of_contours_drawn+1
                else:

                    if (BLUR_STATUS == VERY_BLURRY):
                        other_chk = np.array([sat_chk,long_shape,True])
                        if (border_chk and other_chk.sum() >2):
                            #print ct, other_chk
                            cv2.drawContours(mask3, contours, ct, 1,-1)
                            number_of_contours_drawn = number_of_contours_drawn+1

                    elif (BLUR_STATUS == A_LITTLE_BLUR):
                        if (border_chk and other_chk.sum() >1):
                            #print ct, other_chk
                            cv2.drawContours(mask3, contours, ct, 1,-1)
                            number_of_contours_drawn = number_of_contours_drawn+1

                    else: # BLUR_STATUS = VERY_CLEAR
                        other_chk = np.array([sat_chk,long_shape,edge_chk])
                        if (border_chk and other_chk.sum() >2):
                            #print ct, other_chk
                            cv2.drawContours(mask3, contours, ct, 1,-1)
                            number_of_contours_drawn = number_of_contours_drawn+1

                iteration = iteration + 1

                if number_of_contours_drawn == 0:   
                    relax_constraint = True

        # (7) generate segmentation result
        img_small1 = img_small1*mask3[:,:,np.newaxis]
        img_grabcut1 = cv2.resize(img_small1, None, fx=1.0/scale, fy=1.0/scale, interpolation = cv2.INTER_NEAREST)
        mask4 = cv2.resize(mask3, (n_cols, n_rows), interpolation = cv2.INTER_NEAREST)

        whole_segment_result = np.zeros((image.shape[0],image.shape[1]),dtype=np.int)
        whole_segment_result[r_start:r_start+n_rows,c_start:c_start+n_cols] = 255*mask4

        # (8) Save result
        cv2.imwrite(os.path.join(saveimag_path, title + ext), whole_segment_result) #(saveimag_path+'/frame'+ str(index) +'.jpg',whole_segment_result)
        print (title +" : done! (blur lvl "+str(BLUR_STATUS)+ ", n_contours "+str(number_of_contours_drawn)+")")
        
        
        """fig2 = plt.figure()
        ax1 = fig2.add_subplot(131)
        plt.imshow(img)
        plt.title("Original Image")
        ax2 = fig2.add_subplot(132)
        plt.imshow(whole_segment_result,cmap = 'Greys')
        plt.title("Segmented Image")
        fig2.tight_layout()
        fig2.savefig(saveimag_path+'/frame'+index+'.jpg', bbox_inches = 'tight',pad_inches = 0.1)
        print " frame "+index+" : done! (blur lvl "+str(BLUR_STATUS)+ " blur value:" + str(blur_value)+\
        ", n_contours "+str(number_of_contours_drawn)+")"  """  
        #plt.close()





