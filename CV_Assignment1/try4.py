import numpy as np
import pylab as pl
import skimage
import skimage.io
import skimage.transform
import os
import sys
import tools as t

#----------------------------------------------------------initialize
inputname = t.filepath("siftdata/mandrill.jpg")



#only take grayscale image
Snum = 4 # number of different scales:4
Gnum = 6 # number of Gaussian layers in each scale:6
DoGnum = Gnum - 1
   
GaussianPyramid = {(0,0):0} #initialize a dictionary

#----------------------------------------------------------Gaussian pyramid



I = skimage.img_as_float(skimage.io.imread(inputname))
#IG = skimage.img_as_float(skimage.io.imread(inputname)) # when I is a gray scale image, the reading file is a 2d array
#IG = t.GrayToArray(I)
IG = t.TurnGray(I)

#skimage.io.imsave("siftdata/Gbuilding.jpg", IG)

#------------------------------------------------------------Gaussian pyramid


    
#temp = t.GenerateGaussianLayers(t.ScaleImage2(IG, 2), Gnum)
#
#temp = t.GenerateGaussianLayers(IG, Gnum)   #debug
#
#
#for j in range(Gnum):
#    GaussianPyramid[(0,j)] = temp[j]
# 
#for i in range(1, Snum):
#    temp = t.GenerateGaussianLayers(t.ScaleImage2(GaussianPyramid[(i - 1, Gnum - 3)], 1), Gnum)
#    for j in range(Gnum):
#        GaussianPyramid[(i,j)] = temp[j]
# 
#t.SavePyramid(inputname, GaussianPyramid, "G", Snum, Gnum)



#----------------------------------------------------------DoG

GaussianPyramid = t.ReadInPyramid(inputname, "G", Snum, Gnum)
# 
DoGPyramid = t.DiffofGaussian(GaussianPyramid, Snum, Gnum)
#
#DoGDis = t.DoGPyramidDisplay(DoGPyramid, Snum, Gnum)
#
#t.SavePyramid(inputname, DoGDis, "DoG", Snum, Gnum - 1)
#  

#----------------------------------------------------------Extrema 
 
#Extrema = t.ExtractDoGExtrema(DoGPyramid, Snum, DoGnum)
#
#t.SavePyramid(inputname, Extrema, "ExRaw", Snum, DoGnum - 2)

Extrema = t.ReadInPyramid(inputname, "ExRaw", Snum, DoGnum - 2)
 
t.RefineExtrima(IG, Extrema, DoGPyramid, Snum, Gnum, 10, 9, 0.018)
 
t.SavePyramid(inputname, Extrema, "ExFine", Snum, DoGnum - 2)


# #----------------------------------------------------------read in data

FineExtrema = t.ReadInPyramid(inputname, "ExFine", Snum, Gnum - 3)

ExStack = t.ExtremaLocations(FineExtrema, Snum, Gnum - 3) # a stack of extrema info: scale, layer, x, y.
t.SaveExStack(ExStack, "ExStack3.txt")

ExStack = t.ReadExStack("ExStack3.txt")

MarkedImage = t.DisplayKeyPoints(ExStack, GaussianPyramid, I)

#
skimage.io.imsave("result3.jpg", MarkedImage)
#
#skimage.io.imshow(MarkedImage)
#result = t.GenerateDescriptors(ExStack, GaussianPyramid)
#
#print 4

  
































# Gaussian = t.GenerateGaussianLayers(I, 5)
# I_stack = np.hstack((Gaussian[0], Gaussian[1], Gaussian[2], Gaussian[3], Gaussian[4]))
# #I_stack = np.hstack((Gaussian[0], Gaussian[1]))
# skimage.io.imsave('GS.png', I_stack)

# 
# I = skimage.img_as_float(skimage.io.imread("mandrill.jpg"))
# IG = t.TurnGray(I)
# 
# #only take grayscale image
# Snum = 6 # number of different scales
# Gnum = 6 # number of Gaussian layers in each scale
# 
# Scales = t.GenerateScales(IG, Snum)
# # for i in range(Snum):
# #     fname = "test%d.jpg"  % i
# #     skimage.io.imsave(fname, Scales[i])
#    
#  
#  
# GaussianPyramid = {(0,0):0} #initialize a dictionary
#  
# for i in range(Snum):
#     temp = t.GenerateGaussianLayers(Scales[i], Gnum)
#     for j in range(Gnum):
#         GaussianPyramid[(i,j)] = temp[j]
#         fname = "mandrill%d.jpg"  % i
#         skimage.io.imsave(fname, Scales[i])
# 
# DoGPyramid = t.DiffofGaussian(GaussianPyramid, Snum, Gnum)
#  
# for i in range(Snum):
#     for j in range(Gnum - 1):
#         fname = "DOGmandrill%d%d.jpg"  % (i, j)
#         skimage.io.imsave(fname, DoGPyramid[(i,j)])
