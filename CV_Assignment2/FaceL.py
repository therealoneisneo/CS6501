import numpy as np
import pylab as pl
import skimage
import skimage.io
import skimage.transform
import os
import sys
import tools as t
import FaceData as fd
from numpy.dual import norm


Faces = np.zeros([100, 12, 12, 4])  #all 100 faces
NonFaces = np.zeros([100, 12, 12, 4])  #all 100 non-faces

fd.ReadInFaces(Faces, NonFaces)

FacesVec = fd.UnrollFaces(Faces)# label as 1

NonFacesVec = fd.UnrollFaces(NonFaces)# label as 0

tempFV = np.zeros([FacesVec.shape[0], FacesVec.shape[1] + 1])

tempNFV = np.zeros([NonFacesVec.shape[0], NonFacesVec.shape[1] + 1])

for i in range(FacesVec.shape[0]):
    for j in range(FacesVec.shape[1]):
        tempFV[i, j] = FacesVec[i, j]
for i in range(NonFacesVec.shape[0]):
    for j in range(NonFacesVec.shape[1]):
        tempNFV[i, j] = NonFacesVec[i, j]

for i in range(FacesVec.shape[0]):
    tempFV[i, FacesVec.shape[1]] = 1
for i in range(NonFacesVec.shape[0]):
    tempNFV[i, NonFacesVec.shape[1]] = 1
    
FacesVec = tempFV
NonFacesVec = tempNFV

# w = np.zeros([10, 145])
# for i in range(10):
#     w[i] = fd.GenerateW(FacesVec, NonFacesVec, float(i + 1)/10, 150, 0.5)

 
w = fd.GenerateW(FacesVec, NonFacesVec, 0.1, 150, 0.5)





if (sys.platform == 'linux2'):
    path = "Testset/PhotoCollege.png"
else:
    path = "Testset\\PhotoCollege.png"
    
# copys = 10
#    
#
  
  
# 
# detection = fd.DetectFaceL(path, w)
# outpath = path.split(".")
# output = outpath[0] + "-Lresult.png"
# skimage.io.imsave(output, detection)
#  



 
 
if (sys.platform == 'linux2'):
    path = "Testset/input0.jpg"
else:
    path = "Testset\\input0.jpg"
   
copys = 10
   
   
pathes = fd.CreateGPyramid(path, copys, 0.05)
 
for i in range(copys):
    print i
    detection = fd.DetectFaceL(pathes[i], w)
    outpath = path.split(".")
    output = outpath[0] + "-Lresult-" + str(i) + ".png"
    skimage.io.imsave(output, detection)
a = 0






#  
#     
# result = np.zeros([10, 2, 200])
# correct = np.zeros([10,3])
# for lr in range(10):
#     for i in range(200):
#         InputVec = np.zeros(145)
#         InputVec[144] = 1
#         if (sys.platform == 'linux2'):
#             path = "Testset/test" + str(i) + ".png"
#         else:
#             path = "Testset\\test" + str(i) + ".png"
#         testimage = skimage.img_as_float(skimage.io.imread(path))
#         Input = fd.UnrollImage(testimage)
#         
#         for j in range(144):
#              InputVec[j] = Input[j]
#         wx = np.dot(w[lr], InputVec)
#         g = 1/(1 + np.power(np.e, -wx))
#         result[lr, 0, i] = g
#         if (i <= 78 and g > 0.5):
#             result[lr, 1, i] = 1
#             correct[lr, 0] += 1
#             correct[lr, 2] += 1
#         if (i > 78 and g <= 0.5):
#             result[lr, 1, i] = 1
#             correct[lr, 1] += 1
#             correct[lr, 2] += 1
#         
#     correct[lr, 2] = float(correct[lr, 2])/2

# if (sys.platform == 'linux2'):
#     path = "Testset/group1.png"
# else:
#     path = "Testset\\group1.png"
#   
#   
# result = fd.DetectFaceL(path, w)
# 
# pl.imshow(result)
# pl.show()


