import numpy as np
import pylab as pl
import skimage
import skimage.io
import skimage.transform
import os
import tools as t
import sys

 
# output = open("ImagesList3.txt", 'w')
# # output.write("size " + str(height) + " " + str(width) + "\n")
# for i in range(6532, 6814): 
#     output.write("IMG_" + str(i) + ".JPG\n")
# output.close()
#  

# output = open("ImagesListSy.txt", 'w')
# # output.write("size " + str(height) + " " + str(width) + "\n")
# 
# for i in range(1, 1101): 
# #     print %04d + ".png\n"  % i
#     output.write( "%04d"%i + ".png\n" )  
# #     print ("%04d"%i + ".png\n") 
# output.close()




# Images = t.ReadImages("ImagesList.txt")
#    
# ColorDisM = t.ColorDistanceMatrix(Images)
# # GrayDisM = t.GrayDistanceMatrix(Images)
#   
# t.SaveMatrix("distance.txt", ColorDisM)

# ColorDisM = t.ReadDisMatrix("distance.txt")
# t.SaveMatrix2CSV("distance.csv", ColorDisM)
# ColorDisM = t.ReadDisMatrix("distance2.txt")
# t.SaveMatrix2CSV("distance2.csv", ColorDisM)
# ColorDisM = t.ReadDisMatrix("distance3.txt")
# t.SaveMatrix2CSV("distance3.csv", ColorDisM)

a = 0