#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 00:35:26 2020

@author: qgao
"""
import config
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

def gen_crater_mask(image, cpos, radius, sigma):
    #y,x = np.ogrid[1:shape[0]+1,1:shape[1]+1]
    #image=cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    shape=image.shape
    y,x = np.ogrid[1:shape[0]+1,1:shape[1]+1]
    x = x-cpos[0]
    y = y-cpos[1]
    dist = np.sqrt(x*x+y*y)
    h = np.exp(-(dist-radius)**2/(2.*sigma**2))
     
    maxh = h.max()
    if maxh != 0:
        h /= maxh

    eps = config.eps
    h[h<eps] = eps
    
    #h = h.resize((17,17),Image.ANTIALIAS)
    h2 =cv2.resize(h,(17,17),interpolation=cv2.INTER_AREA) 
    i2= cv2.resize(image,(17, 17),interpolation=cv2.INTER_AREA) 

    r = (i2 * h2).clip(0, 255).astype(np.uint8)
    #r3 = (image * h).clip(0, 255).astype(np.uint8)
    r2= cv2.resize(r,(shape[0], shape[1]),interpolation=cv2.INTER_CUBIC) 
    '''    
    maxr2 = r2.max()
    
    if maxr2 != 0:
        r3 =r2/ maxr2
        
    eps2 = 0.1
    r3[r2<eps2] = eps2
    '''
    #maxh = h.max()
    #if maxh != 0:
    #    h /= maxh

    #eps = 0.2
    #h[h<eps] = eps
    
    h2=h2*255

    
    
    return r2
    

if __name__ == "__main__":
    im = cv2.imread('/home/ran/Trails/PhC-C2DH-U373/CODE/01/resultresponse/v0/49.jpg')
    im=cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    f = gen_crater_mask(im,(141,138), 8, 36.256)
    
    plt.imshow(f)

    cv2.imwrite('/home/ran/Desktop/responsemap/11.png',f)
    
    
    
    #i2= cv2.resize(image,(17, 17),interpolation=cv2.INTER_CUBIC) 
    #i3=cv2.resize(i2,(272, 272),interpolation=cv2.INTER_CUBIC) 
    #r = (i2 * h).clip(0, 255).astype(np.uint8)
    
    #r2= cv2.resize(r,(272, 272)) 
    '''    
    maxr2 = r2.max()
    
    if maxr2 != 0:
        r3 =r2/ maxr2
        
    eps2 = 0.1
    r3[r2<eps2] = eps2
    '''
    #maxh = h.max()
    #if maxh != 0:
    #    h /= maxh

    #eps = 0.2
    #h[h<eps] = eps
    
    #h2=h2*255


