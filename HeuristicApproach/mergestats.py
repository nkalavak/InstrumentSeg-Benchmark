#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

# Look-Up Table for Different Segmentation Parts:
# -------------------------------------------------
#    0   Tissue
#   10   shaft


# Look-Up Table for Different Instrument Types:
# -------------------------------------------------
#   1   Suction
#   2   Electrocautery
#   3   Grasper
#   4   Cutterfor t in range(n_types):
#   5   Bipolar


# In[2]:


#Definition of constant values
n_images = 1706 #700# number of image frames in each video file
n_colors = 256 # number of color steps in each of RGB channels and SV channels in HSV
n_hues = 180   # number of H values in HSV
n_types = 8    # number of instument types

r_start = 0
c_start = 0
n_rows = 475
n_cols = 520

x_label_hsv_2= 'Saturation'
x_label_hsv_1 = 'Hue'
x_label_Oppo_2 = 'Oppo 2'
x_label_Oppo_1 = 'Oppo 1'
y_label = 'Prob (%)'


# In[3]:


# Set path for finding and saving the images
finddata_path = r'/home/nivii/Desktop/MICCAI2019_Dataset/Train/surgery'
image_location = 'Images/'
savedata_path = r'/home/nivii/Desktop/MICCAI2019_Dataset/Train/TotalStats'
savedata_path_merge = r'/home/nivii/Desktop/MICCAI2019_Dataset/Train/MergedStats'
#saveimag_path = r'/home/nivii/Desktop/Data/wicv/
ann_location = 'Annotations/'

extension = r'*.jpg'

#savedata_path = 'G:/Catalyst_March_Quals5/Test/data_20instruments'#'D:/01 Projects/AMAZON CATALYST PROJECT/Quals4/Test/data_20instruments'
#savedata_path_merge = 'G:/Catalyst_March_Quals5/Test/TotalStats_300instruments'#'D:/01 Projects/AMAZON CATALYST PROJECT/Quals4/Test/TotalStats_300instruments'

# load all the files
print ('Start loading statistics data...')
bg_prob_hsv = np.load(savedata_path+'/bg_prob_hsv.npy') # shape:(n_hues, n_colors, n_colors)
instr_prob_hsv = np.load(savedata_path+'/instr_prob_hsv.npy') # shape:(n_types, n_hues, n_colors, n_colors)

bg_prob = np.load(savedata_path+'/bg_prob.npy')         # shape:(n_colors, n_colors, n_colors)
instr_prob = np.load(savedata_path+'/instr_prob.npy')         # shape:(n_types, n_colors, n_colors, n_colors)

print ('Finished loading statistics from files.\n')      


# In[4]:


# Merge the two results
print ('Start merging statistics data...')
bg_prob = bg_prob
instr_prob[0,:,:,:] = instr_prob[0,:,:,:]                          # for tool type 1
instr_prob[1,:,:,:] = instr_prob[1,:,:,:]  
instr_prob[2,:,:,:] = instr_prob[2,:,:,:]                          
instr_prob[3,:,:,:] = instr_prob[3,:,:,:] 
instr_prob[4,:,:,:] = instr_prob[4,:,:,:]                         
instr_prob[5,:,:,:] = instr_prob[5,:,:,:]  
instr_prob[6,:,:,:] = instr_prob[6,:,:,:]  
instr_prob[7,:,:,:] = instr_prob[7,:,:,:]  
print ('Finished merging statistics.\n')


# In[5]:


# Start processing
print ('Start processing statistics data...')
bg_prob_oppo  = np.zeros((n_colors, n_colors))
instr_prob_oppo  = np.zeros((n_types, n_colors, n_colors))
bg_prob_hs  = np.zeros((n_hues, n_colors))
instr_prob_hs  = np.zeros((n_types, n_hues, n_colors))


bg_prob_hs = 100*np.sum(bg_prob_hsv,axis=2)
for tt in range(n_types):
    instr_prob_hs[tt,:,:]  = 100*np.sum(instr_prob_hsv[tt,:,:,:],axis=2)

for r in range(n_colors):
    for g in range(n_colors):
        for b in range (n_colors):
            Oppo1 = int((g-r+255)/2)
            Oppo2 = int((b-(g+r)+2*255)/3)
            
            bg_prob_oppo[Oppo1,Oppo2]  += 100*bg_prob[r,g,b]
            for tt in range (n_types):
                instr_prob_oppo[tt,Oppo1,Oppo2]  += 100*instr_prob[tt,r,g,b]

            
print ('Finished processing statistics.\n')


# In[6]:


# Save the final result of merged statistcs
print ('Save the merged statistics.\n')
np.save(savedata_path_merge+'/bg_prob_oppo.npy', bg_prob_oppo)
np.save(savedata_path_merge+'/sh_prob_oppo.npy', instr_prob_oppo)


np.save(savedata_path_merge+'/bg_prob_hs.npy', bg_prob_hs)
np.save(savedata_path_merge+'/sh_prob_hs.npy', instr_prob_hs)


print ('Finished saving the merged statistics.')


# 
