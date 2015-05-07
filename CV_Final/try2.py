import numpy as np
import pylab as pl
import skimage
import skimage.io
import skimage.transform
import os
import tools as t
import sys

#  
# output = open("ImagesList3.txt", 'w')
# # output.write("size " + str(height) + " " + str(width) + "\n")
# for i in range(6532, 6814): 
#     output.write("IMG_" + str(i) + ".JPG\n")
# output.close()
 
Images = t.ReadImages("ImagesList2.txt")
   
ColorDisM = t.ColorDistanceMatrix(Images)
# GrayDisM = t.GrayDistanceMatrix(Images)
  
t.SaveMatrix("distance2.txt", ColorDisM)

