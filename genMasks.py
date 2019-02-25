# generate masks (gt) from .xml files
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import glob

for filename in glob.glob('TestSet-Annotations/*.xml'):

    tree = ET.parse(filename)
    root = tree.getroot()

    nrows = int( root.find('imagesize')[0].text )
    ncols = int( root.find('imagesize')[1].text )
    mask = np.zeros((nrows, ncols), dtype=np.uint8)

    for obj in root.findall('object'):
        for polygon in obj.findall('polygon'):
            verList = []
            for pt in polygon.findall('pt'):
                x = pt.find('x').text
                y = pt.find('y').text
                verList.append((x, y))
            verList = np.array(verList, np.int32)
            cv2.fillPoly(mask, [verList], 255)

    cv2.imwrite('TestSet-Masks/'+filename[20:-4]+'_masks.jpg', mask)