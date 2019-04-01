#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
import xml.etree.ElementTree as ET
#from shapely.geometry import Polygon, Point
import matplotlib.path as mplPath
# Look-Up Table for Different Segmentation Parts:
# -------------------------------------------------
#    0   Tissue
#   10   shaft


# Look-Up Table for Different Instrument Types:
# -------------------------------------------------
#   1   Suction
#   2   Electrocautery
#   3   Grasper
#   4   Cutter
#   5   Bipolar




# image_path = 'G:/Catalyst_March_Quals5/ColorStats_ToLoad'#'D:/01 Projects/AMAZON CATALYST PROJECT/Quals3/ColorStats_ToLoad'
# stats_path = 'G:/Catalyst_March_Quals5/Test/data_20instruments'#'D:/01 Projects/AMAZON CATALYST PROJECT/Quals4/Test/data_20instruments'
# #XML Parsing using ElementTree
# xml_path = 'G:/Catalyst_March_Quals5/xml_400instruments'#'D:/01 Projects/AMAZON CATALYST PROJECT/Quals4/xml_100instruments'

basedata_path = r'/home/nivii/Desktop/MICCAI2019_Dataset/Train/surgery'#r'/home/nivii/Desktop/Data/wicv/'
image_location = 'Images/'
savedata_path = r'/home/nivii/Desktop/MICCAI2019_Dataset/Train/TotalStats'
#saveimag_path = r'/home/nivii/Desktop/Data/wicv/
ann_location = 'Annotations/'

extension = r'*.jpg'
#Initialize values
n_colors = 256
n_hues = 180
n_types = 8
r_start = 0
c_start = 0
n_rows = 475
n_cols = 520
n_images = 1706 #700 #300

tissue_prob = np.zeros((n_colors, n_colors, n_colors))
instr_prob = np.zeros((n_types,n_colors, n_colors, n_colors))

tissue_prob_hsv = np.zeros((n_hues, n_colors, n_colors))
instr_prob_hsv = np.zeros((n_types,n_hues, n_colors, n_colors))


# In[2]:


bg_count = 0
instr_count = [0,0,0,0,0,0,0,0] 

bg_occurance = np.zeros((n_colors, n_colors, n_colors))
instr_occurance = np.zeros((n_types, n_colors, n_colors, n_colors))

bg_occurance_hsv = np.zeros((n_hues, n_colors, n_colors))
instr_occurance_hsv = np.zeros((n_types, n_hues, n_colors, n_colors))


# In[4]:


for i in range(1,7):
    finddata_path = basedata_path + str(i)
    print("Inside Folder Surgery: ", i)

    for pathAndFilename in glob.iglob(os.path.join(finddata_path, image_location, extension)):
        #index = '%03d' % img
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))
        orig_image = cv2.imread(os.path.join(finddata_path, image_location, title + ext))
        print("Extracting image: ", title+ext)   

        """rows,cols,_ = orig_image.shape
        new_image1 = cv2.flip(orig_image,0)
        M = cv2.getRotationMatrix2D((cols/2,rows/2),-90,1)
        final_image = cv2.warpAffine(new_image1, M, (cols,rows))
        orig_image = final_image"""
        orig_image_hsv = cv2.cvtColor(orig_image,cv2.COLOR_BGR2HSV)

        curr_label = np.asarray(cv2.imread(os.path.join(finddata_path, image_location, title + ext)))
            #image_path  + '/frame' + str(index) + '.jpg'))
        curr_label = curr_label[:,:,0]


        tree = ET.parse(os.path.join(finddata_path, ann_location, title + r'.xml'))#xml_path + '/frame' + str(index) + '.xml')
        root = tree.getroot()
        #Initializing values before evaluating each image
        x = []
        y = []
        poly1_s = None
        poly1_a = None
        poly2_l = None
        poly3_l = None
        poly4_l = None
        poly5_l = None
        poly6 = None
        poly2_r = None
        poly3_r = None
        poly4_r = None
        poly5_r = None
        poly7 = None
        poly8 = None

        #Looks for <object> tag in xml file
        for neighbor in root.iter('object'):      
            #print neighbor.find('name').text
            for charac in neighbor.findall('polygon'):
                #Save the points inside <polygon>
                points = []           
                for verts in charac.iter('pt'):
                    y.append(int(verts.find('x').text))
                    x.append(int(verts.find('y').text)) 
                points = zip(x,y)

                #Name polygons from points
                if(neighbor.find('name').text == 'suction' and (neighbor.find('attributes').text == 'surgeon suction' or neighbor.find('attributes').text == None)):
                    poly1_s = mplPath.Path(list(points))
                if(neighbor.find('name').text == 'suction' and neighbor.find('attributes').text == 'assistant suction'):
                    poly1_a = mplPath.Path(list(points))
                if(neighbor.find('name').text == 'electrocautery' and (neighbor.find('attributes').text == 'left cautery' or neighbor.find('attributes').text == None)):
                    poly2_l = mplPath.Path(list(points))
                if(neighbor.find('name').text == 'electrocautery' and neighbor.find('attributes').text == 'right cautery'):
                    poly2_r = mplPath.Path(list(points))
                if(neighbor.find('name').text == 'grasper' and (neighbor.find('attributes').text == 'left grasper' or neighbor.find('attributes').text == None)):
                    poly3_l = mplPath.Path(list(points))
                if(neighbor.find('name').text == 'grasper' and neighbor.find('attributes').text == 'right grasper'):
                    poly3_r = mplPath.Path(list(points))
                if(neighbor.find('name').text == 'cutter' and (neighbor.find('attributes').text == 'left cutter' or neighbor.find('attributes').text == None)):
                    poly4_l = mplPath.Path(list(points))
                if(neighbor.find('name').text == 'cutter' and neighbor.find('attributes').text == 'right cutter'):
                    poly4_r = mplPath.Path(list(points))
                if(neighbor.find('name').text == 'pickup' and (neighbor.find('attributes').text == 'left pickup' or neighbor.find('attributes').text == None)):
                    poly5_l = mplPath.Path(list(points))
                if(neighbor.find('name').text == 'pickup' and neighbor.find('attributes').text == 'right pickup'):
                    poly5_r = mplPath.Path(list(points))
                if(neighbor.find('name').text ==  'curette'):
                    poly6 = mplPath.Path(list(points))
                if(neighbor.find('name').text == 'drill'):
                    poly7 = mplPath.Path(list(points))
                if(neighbor.find('name').text == 'others'):
                    poly8 = mplPath.Path(list(points))
                #Clear list of x and y
                x[:] = []
                y[:] = []
                #print poly2_l, poly2_r, poly3_l, poly3_r, poly4_l, poly4_r, poly5_l, poly5_r, poly6, poly7

            #Check location of each pixel based on location of polygon
            for r in range(r_start, r_start + n_rows):
                for c in range(c_start, c_start + n_cols):
                    pixel_color = orig_image[r,c]
                    pixel_color_hsv = orig_image_hsv[r,c]
                    pixel_color = [int(pixel_color[0]), int(pixel_color[1]), int(pixel_color[2])]
                    pixel_color_hsv = [int(round(pixel_color_hsv[0])), int(round(pixel_color_hsv[1])), int(round(pixel_color_hsv[2]))]    
                    #Check if point is present inside suction mask
                    if((poly1_a != None and poly1_a.contains_point((r,c)) == True) or (poly1_s != None and poly1_s.contains_point((r,c)) == True)):
                        instr_count[0] += 1
                        instr_occurance[0, pixel_color[0], pixel_color[1], pixel_color[2]] +=1
                        instr_occurance_hsv[0,pixel_color_hsv[0],pixel_color_hsv[1],pixel_color_hsv[2]] += 1

                    elif((poly2_l != None and poly2_l.contains_point((r,c)) == True) or (poly2_r != None and poly2_r.contains_point((r,c)) == True)):
                        instr_count[1] += 1
                        instr_occurance[1, pixel_color[0], pixel_color[1], pixel_color[2]] +=1
                        instr_occurance_hsv[1,pixel_color_hsv[0],pixel_color_hsv[1],pixel_color_hsv[2]] += 1

                    elif((poly3_l != None and poly3_l.contains_point((r,c)) == True) or (poly3_r != None and poly3_r.contains_point((r,c)) == True)):
                        instr_count[2] += 1
                        instr_occurance[2, pixel_color[0], pixel_color[1], pixel_color[2]] +=1
                        instr_occurance_hsv[2,pixel_color_hsv[0],pixel_color_hsv[1],pixel_color_hsv[2]] += 1
                    elif((poly4_l != None and poly4_l.contains_point((r,c)) == True) or (poly4_r != None and poly4_r.contains_point((r,c)) == True)):
                        instr_count[3] += 1
                        instr_occurance[3, pixel_color[0], pixel_color[1], pixel_color[2]] +=1
                        instr_occurance_hsv[3,pixel_color_hsv[0],pixel_color_hsv[1],pixel_color_hsv[2]] += 1
                    elif((poly5_l != None and poly5_l.contains_point((r,c)) == True) or (poly5_r != None and poly5_r.contains_point((r,c)) == True)):
                        instr_count[4] += 1
                        instr_occurance[4, pixel_color[0], pixel_color[1], pixel_color[2]] +=1
                        instr_occurance_hsv[4,pixel_color_hsv[0],pixel_color_hsv[1],pixel_color_hsv[2]] += 1
                    elif(poly6 != None and poly6.contains_point((r,c)) == True):
                        instr_count[5] += 1
                        instr_occurance[5, pixel_color[0], pixel_color[1], pixel_color[2]] +=1
                        instr_occurance_hsv[5,pixel_color_hsv[0],pixel_color_hsv[1],pixel_color_hsv[2]] += 1
                    elif(poly7 != None and poly7.contains_point((r,c)) == True):
                        instr_count[6] += 1
                        instr_occurance[6, pixel_color[0], pixel_color[1], pixel_color[2]] +=1
                        instr_occurance_hsv[6,pixel_color_hsv[0],pixel_color_hsv[1],pixel_color_hsv[2]] += 1
                    elif(poly8 != None and poly8.contains_point((r,c)) == True):
                        instr_count[7]+= 1
                        instr_occurance[7, pixel_color[0], pixel_color[1], pixel_color[2]] +=1
                        instr_occurance_hsv[7, pixel_color_hsv[0],pixel_color_hsv[1],pixel_color_hsv[2]] += 1

                    else:
                        bg_count +=1
                        bg_occurance[pixel_color[0], pixel_color[1], pixel_color[2]] +=1
                        bg_occurance_hsv[pixel_color_hsv[0],pixel_color_hsv[1],pixel_color_hsv[2]] +=1


# In[5]:


print ('Number of background pixels: ' + str(bg_count))
print ('Number of shaft pixels:' + str(instr_count))


# In[7]:


np.save(savedata_path+'/bg_occurance.npy', bg_occurance)
np.save(savedata_path+'/instr_occurance.npy', instr_occurance)
np.save(savedata_path+'/bg_occurance_hsv_dataset.npy', bg_occurance_hsv)
np.save(savedata_path+'/instr_occurance_hsv.npy', instr_occurance_hsv)
np.save(savedata_path+'/count_background.npy', np.array([bg_count]))
np.save(savedata_path+'/count_toolparts.npy', np.array([instr_count]))
    


# In[8]:


# load to probability
tissue_prob += bg_occurance/float(bg_count)
tissue_prob_hsv += bg_occurance_hsv/float(bg_count)

for t in range(n_types):
    if instr_count[t] > 0:
        normalize_factor = float(instr_count[t])
        if normalize_factor != 0:
            instr_prob[t,:,:,:] += instr_occurance[t,:,:,:]/normalize_factor
            instr_prob_hsv[t,:,:,:] += instr_occurance_hsv[t,:,:,:]/normalize_factor
            #print instr_prob[t,:,:,:]
        else:
            print ('Probability calculation error: type'+str(t)+' instr_count='+str(instr_count[t]))


# In[10]:


print ('Start saving statistics data...')
np.save(savedata_path+'/bg_prob.npy', tissue_prob)
np.save(savedata_path+'/instr_prob.npy', instr_prob)
np.save(savedata_path+'/bg_prob_hsv.npy', tissue_prob_hsv)
np.save(savedata_path+'/instr_prob_hsv.npy', instr_prob_hsv)

print ('Finished saving statistics to file. Goodbye!')

print ('Sanity check:'+str(instr_prob[:].max()))


# In[22]:


print ("Instrument Prob:",instr_prob[6].sum())






