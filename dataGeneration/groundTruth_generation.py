# Written by Niveditha Kalavakonda (nkalavak@uw.edu)
# Fall 2017 - Dataset Generation

import numpy as np
import matplotlib.pyplot as plt
import cv2
import imutils
import xml.etree.ElementTree as ET
#from shapely.geometry import Polygon, Point
import matplotlib.path as mplPath


# Look-Up Table for Different Segmentation Parts:
# -------------------------------------------------
#    0   Instrument 
#   255  Tissue


# Look-Up Table for Different Instrument Types:
# -------------------------------------------------
#   1   Suction
#   2   Electrocautery
#   3   Grasper
#   4   Cutter
#   5   Bipolar

image_path = 'D:/01 Projects/AMAZON CATALYST PROJECT/ColorStats_ToLoad'

save_path = 'D:/01 Projects/AMAZON CATALYST PROJECT/Dataset'

#XML Parsing using ElementTree
xml_path = 'D:/01 Projects/AMAZON CATALYST PROJECT/xml_100instruments'

r_start = 0
c_start = 0
n_rows = 480 #475
n_cols = 720 #520
n_images = 20 #100

background = [0, 0, 0]
instr = [255, 255, 255]
start_image_index = 70 #300


for img in range(start_image_index, start_image_index + n_images):
#img = 14
    index = img #'%03d' % img
    orig_image = cv2.imread(image_path + '/frame' + str(index) + '.jpg')

    #To rotate and flip image

    #new_image1 = cv2.flip(orig_image)
    ##final_image = imutils.rotate_bound(new_image1,90)

    """rows,cols,_ = orig_image.shape
    new_image1 = cv2.flip(orig_image,0)
    M = cv2.getRotationMatrix2D((cols/2,rows/2),-90,1)
    final_image = cv2.warpAffine(new_image1, M, (cols,rows))"""
    #cv2.imshow("Flipped Image", final_image)
    #print (orig_image.shape)
    #print (final_image.shape)

    new_image = np.zeros(orig_image.shape)

    #n_rows,n_cols = n_cols,n_rows

    tree = ET.parse(xml_path + '/frame' + str(index) + '.xml')
    root = tree.getroot()
    
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
    for neighbor in root.iter('object'): 
        for charac in neighbor.findall('polygon'):
            points = []  
            for verts in charac.iter('pt'):
                y.append(int(verts.find('x').text))
                x.append(int(verts.find('y').text)) 
            #print (x)
            #print (y)

            points = zip(x,y)
            """if(neighbor.find('name').text == 'suction' and (neighbor.find('attributes').text == 'surgeon suction' or neighbor.find('attributes').text == None)):
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
                poly8 = mplPath.Path(list(points))"""
            if(neighbor.find('name').text == 'surgeon suction' or neighbor.find('name').text == 'suction'):
                poly1_s = mplPath.Path(list(points))
            if(neighbor.find('name').text == 'assistant suction'):
                poly1_a = mplPath.Path(list(points))
            if(neighbor.find('name').text == 'left cautery' or neighbor.find('name').text == 'electrocautery'):
                poly2_l = mplPath.Path(list(points))
            if(neighbor.find('name').text == 'right cautery'):
                poly2_r = mplPath.Path(list(points))
            if(neighbor.find('name').text == 'left grasper' or (neighbor.find('name').text == 'grasper')):
                poly3_l = mplPath.Path(list(points))
            if(neighbor.find('name').text == 'right grasper'):
                poly3_r = mplPath.Path(list(points))
            if(neighbor.find('name').text == 'cutter' or (neighbor.find('name').text == 'left cutter')):
                poly4_l = mplPath.Path(list(points))
            if(neighbor.find('name').text == 'right cutter'):
                poly4_r = mplPath.Path(list(points))
            if(neighbor.find('name').text == 'pickup' or (neighbor.find('name').text == 'left pickup')):
                poly5_l = mplPath.Path(list(points))
            if(neighbor.find('name').text == 'right pickup'):
                poly5_r = mplPath.Path(list(points))
            if(neighbor.find('name').text ==  'curette'):
                poly6 = mplPath.Path(list(points))
            if(neighbor.find('name').text == 'drill'):
                poly7 = mplPath.Path(list(points))
            if(neighbor.find('name').text == 'others'):
                poly8 = mplPath.Path(list(points))

            x[:] = []
            y[:] = []
        for c in range(c_start + n_cols):
            for r in range(r_start + n_rows):

                #Check if point is present inside suction mask
                if((poly1_a != None and poly1_a.contains_point((r,c)) == True) or (poly1_s != None and poly1_s.contains_point((r,c)) == True)):
                    new_image[r,c] = instr#orig_image[r,c]

                elif((poly2_l != None and poly2_l.contains_point((r,c)) == True) or (poly2_r != None and poly2_r.contains_point((r,c)) == True)):
                    new_image[r,c] = instr#orig_image[r,c]#instr#

                elif((poly3_l != None and poly3_l.contains_point((r,c)) == True) or (poly3_r != None and poly3_r.contains_point((r,c)) == True)):
                    new_image[r,c] = instr#orig_image[r,c]#instr

                elif((poly4_l != None and poly4_l.contains_point((r,c)) == True) or (poly4_r != None and poly4_r.contains_point((r,c)) == True)):
                    new_image[r,c] = instr#orig_image[r,c]#instr

                elif((poly5_l != None and poly5_l.contains_point((r,c)) == True) or (poly5_r != None and poly5_r.contains_point((r,c)) == True)):
                    new_image[r,c] = instr#orig_image[r,c]#instr

                elif(poly6 != None and poly6.contains_point((r,c)) == True):
                    new_image[r,c] = instr#orig_image[r,c]#instr

                elif(poly7 != None and poly7.contains_point((r,c)) == True):
                    new_image[r,c] = instr#orig_image[r,c]#instr

                elif(poly8 != None and poly8.contains_point((r,c)) == True):
                    new_image[r,c] = instr#orig_image[r,c]#instr

                else:
                    new_image[r,c] = background

    cv2.imwrite(save_path+'/frame'+str(index)+'.jpg',new_image)
    print ("frame "+ str(index) +" : done!")
