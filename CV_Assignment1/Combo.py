import numpy as np
import pylab as pl
import skimage
import skimage.io
import skimage.transform
import os
import tools as t

I = {0:0}
# I[0] = skimage.img_as_float(skimage.io.imread("testcases\largebuildings.jpg"))
# I[1] = skimage.img_as_float(skimage.io.imread("testcases\\building.jpg"))
# I[2] = skimage.img_as_float(skimage.io.imread("testcases\\mandrill.jpg"))
# I[3] = skimage.img_as_float(skimage.io.imread("testcases\\Iron-Man-in-Mansion.jpg"))
# I[4] = skimage.img_as_float(skimage.io.imread("testcases\\lena.png"))
I[0] = skimage.img_as_float(skimage.io.imread("testcases\\checker.jpg"))

for i in range(len(I)):
    IG = t.Gsmooth(I[i], 2, 7)
    Gray = t.TurnGray(IG)
    fname = "testcases\Gray%d.jpg"  % i
    skimage.io.imsave(fname, Gray)
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
     
    #DT = t.Double_Threshold(INMSmapped,0.65)
    DT = t.Double_Threshold(INMSmapped,0.45)
    fname = "testcases\Dthreshold%d.jpg"  % i
    skimage.io.imsave(fname, DT)
      
      
    CovarianceMatrix = np.zeros([a,b,2,2])
    EigenM = t.GenerateCovarianceEig(CovarianceMatrix, GradientX, GradientY, 7)
     
    CornerCandidate = t.TruncEigenvalue(EigenM, 0.97)
    Corner = t.CornerPlot(CornerCandidate, I[i])
    fname = "testcases\CornerCandi%d.jpg"  % i
    skimage.io.imsave(fname, Corner)
    CornerPixels = t.GenerateCornerPoint(CornerCandidate, a, b, 19)
    Corner = t.CornerPlot(CornerPixels, I[i])
    fname = "testcases\Corners%d.jpg"  % i
    skimage.io.imsave(fname, Corner)

