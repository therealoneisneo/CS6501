import numpy as np
import skimage
import skimage.io
import tools as t
import os


#----------------------------------------------------------initialize
inputname = t.filepath("test/building.jpg")



#only take grayscale image
Snum = 4 # number of different scales:4
Gnum = 6 # number of Gaussian layers in each scale:6
DoGnum = Gnum - 1
   
GaussianPyramid = {(0,0):0} #initialize a dictionary

#----------------------------------------------------------Gaussian pyramid



I = skimage.img_as_float(skimage.io.imread(inputname))
# IG = t.TurnGray(I)
# 
# temp = t.GenerateGaussianLayers(t.ScaleImage2(IG, 2), Gnum)
# for j in range(Gnum):
#     GaussianPyramid[(0,j)] = temp[j]
#  
# for i in range(1, Snum):
#     temp = t.GenerateGaussianLayers(t.ScaleImage2(GaussianPyramid[(i - 1, Gnum - 3)], 1), Gnum)
#     for j in range(Gnum):
#         GaussianPyramid[(i,j)] = temp[j]
# 
# 
# t.SavePyramid(inputname, GaussianPyramid, "G", Snum, Gnum)



#----------------------------------------------------------DoG
# 
# GaussianPyramid = t.ReadInPyramid(inputname, "G", Snum, Gnum)
# 
# testdic = {(0,0):0}
# for i in range(Snum):
#     for j in range(1, Gnum):
#         testdic[i, j - 1] = GaussianPyramid[i, j]

rotate = skimage.transform.rotate(I, 65, False, (100, 0))

skimage.io.imshow(I)

skimage.io.imshow(rotate)
#   
# DoGPyramid = t.DiffofGaussian(GaussianPyramid, Snum, Gnum)
#   
# t.SavePyramid(inputname, DoGPyramid, "DoG", Snum, Gnum - 1)
  
  
#----------------------------------------------------------Extrema 
 
# Extrema = t.ExtractDoGExtrema(DoGPyramid, Snum, DoGnum)
#   
# t.SavePyramid(inputname, Extrema, "ExRaw", Snum, DoGnum - 1)
#  
# Extrema = t.RefineExtrima(Extrema, DoGPyramid, Snum, Gnum, 10, 6, 0.02)
#  
# t.SavePyramid(inputname, Extrema, "ExFineA", Snum, DoGnum - 1)
#  
# Extrema = t.RefineExtrima(Extrema, DoGPyramid, Snum, Gnum, 10, 7, 0.02)
#  
# t.SavePyramid(inputname, Extrema, "ExFineB", Snum, DoGnum - 1)
#  
# Extrema = t.RefineExtrima(Extrema, DoGPyramid, Snum, Gnum, 10, 8, 0.02)
#  
# t.SavePyramid(inputname, Extrema, "ExFineC", Snum, DoGnum - 1)
#  
# Extrema = t.RefineExtrima(Extrema, DoGPyramid, Snum, Gnum, 10, 9, 0.02)
#  
# t.SavePyramid(inputname, Extrema, "ExFineD", Snum, DoGnum - 1)
#  
# Extrema = t.RefineExtrima(Extrema, DoGPyramid, Snum, Gnum, 10, 6, 0.01)
#  
# t.SavePyramid(inputname, Extrema, "ExFineF", Snum, DoGnum - 1)
#  
# Extrema = t.RefineExtrima(Extrema, DoGPyramid, Snum, Gnum, 10, 7, 0.01)
#  
# t.SavePyramid(inputname, Extrema, "ExFineG", Snum, DoGnum - 1)
#  
# Extrema = t.RefineExtrima(Extrema, DoGPyramid, Snum, Gnum, 10, 8, 0.01)
#  
# t.SavePyramid(inputname, Extrema, "ExFineH", Snum, DoGnum - 1)
#  
# Extrema = t.RefineExtrima(Extrema, DoGPyramid, Snum, Gnum, 10, 9, 0.01)
#  
# t.SavePyramid(inputname, Extrema, "ExFineI", Snum, DoGnum - 1)
#  
# Extrema = t.RefineExtrima(Extrema, DoGPyramid, Snum, Gnum, 10, 10, 0.01)
#  
# t.SavePyramid(inputname, Extrema, "ExFineE", Snum, DoGnum - 1)


