# In[Libraries]
import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
#Tesseract Library
import pytesseract
import random as rn
import re

from PIL import Image
from scipy.spatial import distance
from scipy import stats



pytesseract.pytesseract.tesseract_cmd = r'/usr/local/Cellar/tesseract/5.3.0_1/bin/tesseract'

def add_padding(img_, padding=20, r=255, g=255, b=255):

    if len(img_.shape) == 2:
        img_ = np.stack((img_,)*3, axis=-1)
        
    old_image_height, old_image_width, channels = img_.shape

    new_height = old_image_height + padding
    new_width =  old_image_width + padding
    
    result = np.full((new_height, new_width, channels), (r,g,b), dtype=np.uint8)
    
    # compute center offset
    x_center = (new_width - old_image_width) // 2
    y_center = (new_height - old_image_height) // 2
    
    # copy crop1 image into center of result image
    result[y_center:y_center+old_image_height, 
           x_center:x_center+old_image_width] = img_
    
    
    return result



def img_2_str(img, config_int=7, special=''):
    '''
    https://pyimagesearch.com/2021/11/15/tesseract-page-segmentation-modes-psms-explained-how-to-improve-your-ocr-accuracy/
    
    Parameters
    ----------
    img : ___
    
    config_int : 
        6: Assume a single uniform block of text
        7: Single Line Text
        12: Sparse Text
        
    Returns string from iamge
    '''       
    text = pytesseract.image_to_string(img, config=f'--psm {config_int} {special}')
    return text.replace('\n', ' ')
    


def bgr_prep(img):
    
    # all cropped keys have a trailing section with useless info
    rectangle = np.zeros((100,50), dtype='uint8')
    img = cv.rectangle(img, (350,150), (460,190), (255,255,255), -1)
    
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # threshold the image using Otsu's thresholding method
    thresh = cv.threshold(gray, 0, 255,
    	cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
    
    # apply a distance transform which calculates the distance to the
    # closest zero pixel for each pixel in the input image
    dist = cv.distanceTransform(thresh, cv.DIST_L2, 5)
    # normalize the distance transform such that the distances lie in
    # the range [0, 1] and then convert the distance transform back to
    # an unsigned 8-bit integer in the range [0, 255]
    dist = cv.normalize(dist, dist, 0, 1.0, cv.NORM_MINMAX)
    dist = (dist * 255).astype("uint8")
     
    # threshold the distance transform using Otsu's method
    dist = cv.threshold(dist, 0, 255,
    	cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    
    dist = np.tile(dist[:, :, None], [1, 1, 3])
    
    return dist


def mser_detect(img_):
    gray = cv.cvtColor(img_, cv.COLOR_BGR2GRAY)
    
    chars = []
    canvas = np.zeros((200,500), dtype='uint8')
    
    (h, w) = gray.shape[:2]
    image_size = h*w
    mser = cv.MSER_create()
    mser.setMaxArea(round(image_size/2))
    mser.setMinArea(2)
    
    _, bw = cv.threshold(gray, 0.0, 255.0, cv.THRESH_OTSU )
    
    regions, rects = mser.detectRegions(bw)
    
    # With the rects you can e.g. crop the letters
    for (x, y, w, h) in rects:
        # cv.rectangle(gray, (x, y), (x+w, y+h), color=(255, 0, 255), thickness=1)
        if w > 15 and h >25:
            cv.rectangle(canvas, (x, y), (x+w, y+h), color=(255, 0, 255), thickness=1)
    
    return canvas


def show(_img, cv_=True):
    if cv_:
        plt.imshow(cv.cvtColor(_img, cv.COLOR_BGR2RGB))
    else:
        plt.imshow(_img)
        


def return_chars(whole_img, x,y,w,h, x_s=390, x_e=700, y_s=80, y_e=110):
    
    # the default x_s, x_e, y_s, y_e only works for single side scans!
    
    # middle and far right scans (sheet 2 & 3) reflect differently & thus
    # a different quadrant of the Windows Logo is highlighted/
    '''
    Parameters
        ----------
        whole_img : whole_scan
        
        x & w: are starting points for the the lower left corner of Windows Logo
        
        wethen 
   
        Returns
        -------
        mask :  which is the text itself
    '''     
     
    # calibrate for final version
    scope = whole_img[y + y_s : y + h + y_e, x + x_s :x + w + x_e]
    
    # all cropped keys have a trailing section with useless info
    rectangle = np.zeros((100,50), dtype='uint8')
    scope = cv.rectangle(scope, (350,120), (480,190), (60,60,60), -1)
    
    # Define lower and uppper limits for the color spectrum of white
    lower = np.array([0, 0, 155])
    upper = np.array([180, 40, 255])
    
    imgHSV = cv.cvtColor(scope, cv.COLOR_BGR2HSV)
    # create the Mask
    mask = cv.inRange(imgHSV, lower, upper)
    # inverse mask
    mask = 255-mask
    
    return mask


# In[]







