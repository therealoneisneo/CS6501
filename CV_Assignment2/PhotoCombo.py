import numpy as np
import pylab as pl
import skimage
import skimage.io
import skimage.transform
import os
import tools as t
import FaceData as fd
import sys





Faces = np.zeros([100, 12, 12, 4])  #all 100 faces
NonFaces = np.zeros([100, 12, 12, 4])  #all 100 non-faces

fd.GenerateFaces("GroundTruth.txt")

fd.GenerateNonFaces(15)

fd.ReadInFaces(Faces, NonFaces)

PhotoCollege = fd.MakeCombo(Faces, NonFaces)
 
PhotoCollege = skimage.transform.rescale(PhotoCollege, 2)


if (sys.platform == 'linux2'):
    path = "Results/PhotoCollege.png"
else:
    path = "Results\\PhotoCollege.png"
 
pl.imshow(PhotoCollege)
pl.show()
 
skimage.io.imsave(path, PhotoCollege)
  
 
