import numpy as np
import pylab as pl
import skimage
import skimage.io
import skimage.transform
import os
import tools as t

I = {0:0}
I[0] = skimage.img_as_float(skimage.io.imread("testcases\largebuildings.jpg"))
I[1] = skimage.img_as_float(skimage.io.imread("testcases\\building.jpg"))
I[2] = skimage.img_as_float(skimage.io.imread("testcases\\mandrill.jpg"))
I[2] = skimage.img_as_float(skimage.io.imread("testcases\\Iron-Man-in-Mansion.jpg"))

#pl.imshow(I)
#pl.show()

IG = t.Gsmooth(I, 1, 5)

Gray = t.TurnGray(IG)

(a,b,c) = I.shape
GradientX = np.zeros([a,b])
GradientY = np.zeros([a,b])
t.ComputeGradient(Gray, GradientX, GradientY)

#GaussianMask = t.Gkernel(1, 5)
CovarianceMatrix = np.zeros([a,b,2,2])
EigenM = t.GenerateCovarianceEig(CovarianceMatrix, GradientX, GradientY, 3)


CornerCandidate = t.TruncEigenvalue(EigenM, 0.94)
Corner = t.CornerPlot(CornerCandidate, I)
fname = "testcases\chessCornerCandi.jpg"  #% i
skimage.io.imsave(fname, Corner)
CornerPixels = t.GenerateCornerPoint(CornerCandidate, a, b, 15)
Corner = t.CornerPlot(CornerPixels, I)
fname = "testcases\chessCorner.jpg"  #% i
skimage.io.imsave(fname, Corner)


#Corner = t.CornerPlot(CornerPoints, Gray)
# Corner = t.CornerPlot(CornerPixels, I)
# Cornergray = t.CornerPlot(CornerPixels, Gray)

 
# 
#   
# I_stack = np.hstack((I, Corner, Cornergray))
#    
# pl.imshow(I_stack)
# pl.show()


