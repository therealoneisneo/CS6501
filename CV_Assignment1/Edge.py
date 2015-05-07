import numpy as np
import pylab as pl
import skimage
import skimage.io
import skimage.transform
import os
import tools as t

I = {0:0}

I[0] = skimage.img_as_float(skimage.io.imread("K:\\UVa HWs\\TA work\\2015spring\\Graphics\\Assignment 1_ Image Processing\\Locascio, Andrew(aml7hp)\\Submission attachment(s)\\aml7hp\\images\\contrast.jpg"))
# I[0] = skimage.img_as_float(skimage.io.imread("testcases\largebuildings.jpg"))
# I[1] = skimage.img_as_float(skimage.io.imread("testcases\\building.jpg"))
# I[2] = skimage.img_as_float(skimage.io.imread("testcases\\mandrill.jpg"))
# I[2] = skimage.img_as_float(skimage.io.imread("testcases\\Iron-Man-in-Mansion.jpg"))
# I[0] = skimage.img_as_float(skimage.io.imread("testcases\\mandrill.jpg"))
#I = skimage.img_as_float(skimage.io.imread('E:/lena_color.png'))
#IG = skimage.img_as_float(skimage.io.imread('E:/lena_color_smoothed.png'))

for i in range(len(I)):
#-----------------------------------------------------------
    IG = t.Gsmooth(I[i], 2, 7)
    Gray = t.TurnGray(IG)
    (a,b,c) = I[i].shape
    GradientM = np.zeros([a,b])
    #IS = t.Sobel(IG, GradientM)
    
    GradientX = np.zeros([a,b])
    GradientY = np.zeros([a,b])
    t.ComputeGradient(Gray, GradientX, GradientY)
    
    fname = "testcases\GradientX%d.jpg"  % i
    skimage.io.imsave(fname, t.Gradient_to_img(GradientX))
    fname = "testcases\GradientY%d.jpg"  % i
    skimage.io.imsave(fname, t.Gradient_to_img(GradientY))
    
    
    IS = t.Sobel(Gray, GradientM)
    ISmapped = t.ScaleMapping(IS)
    
    fname = "testcases\Sobel%d.jpg"  % i
    skimage.io.imsave(fname, ISmapped)
     
    INMS = t.NMS(IS, GradientM)
    INMSmapped = t.ScaleMapping(INMS)
    
      
    fname = "testcases\NMS%d.jpg"  % i
    skimage.io.imsave(fname, INMSmapped)
      
    # DT = t.Double_Threshold(INMSmapped,0.2)
    # DT1 = t.Double_Threshold(INMSmapped,0.4)
    # DT2 = t.Double_Threshold(INMSmapped,0.6)
    DT = t.Double_Threshold(INMSmapped,0.65)
    fname = "testcases\Dthreshold%d.jpg"  % i
    skimage.io.imsave(fname, DT)
    

