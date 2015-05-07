import numpy as np
import pylab as pl
import skimage
import skimage.io
import tools as t
import os
# M1 = np.array([[3,8,1],[9,2,7],[6,5,4]])
# M2 = np.zeros([1,3])
# M2[0] = [1,2,3]
# M3 = [[4,5,6]]
# M2 = np.append(M2, M3, axis=0)
# print np.argsort(M1,axis=0)
 
I = skimage.img_as_float(skimage.io.imread("testcases\checkerRotate.jpg"))


Corner = I
fname = "testcases\chessCornerCand.jpg"  #% i
skimage.io.imsave(fname, Corner)
# pl.imshow(IG)
# pl.show()

# 
# for i in range(1,3):
#     fname = "test%d%d.png"  % (i, i+1)
#     print fname 

# print M2