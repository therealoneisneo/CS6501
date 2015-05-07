import numpy as np
import pylab as pl
import skimage
import skimage.io
import skimage.transform
import os
import tools as t
import FaceData as fd
import sys

# 
# 
# path = "Testset\\group2.jpg"
# 
# fd.CreateGPyramid(path, 10, 0.05)

# fd.GenerateFaces("GroundTruth.txt")

# fd.GenerateNonFaces(15)


Faces = np.zeros([100, 12, 12, 4])  #all 100 faces
NonFaces = np.zeros([100, 12, 12, 4])  #all 100 non-faces

# fd.GenerateFaces("GroundTruth.txt")

# fd.GenerateNonFaces(15)
 
fd.ReadInFaces(Faces, NonFaces)
 
# PhotoCollege = fd.MakeCombo(Faces, NonFaces)
# pl.imshow(PhotoCollege)
# pl.show()
 
mean1 = fd.FaceMean(Faces)
 
mean2 = fd.FaceMean(NonFaces)

mean = np.hstack((mean1,mean2))

if (sys.platform == 'linux2'):
    path = "Results/mean.png"
else:
    path = "Results\\mean.png"
 
mean = skimage.transform.resize(mean, (240, 480))

skimage.io.imsave(path, mean)
