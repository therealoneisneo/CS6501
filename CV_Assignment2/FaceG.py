import numpy as np
import pylab as pl
import skimage
import skimage.io
import skimage.transform
import os
import sys
import tools as t
import FaceData as fd





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
 
FacesVec = fd.UnrollFaces(Faces)
 
NonFacesVec = fd.UnrollFaces(NonFaces)
 
CovFaces = fd.CovMatrix(FacesVec, fd.UnrollImage(mean1))
 
CovNonFaces = fd.CovMatrix(NonFacesVec, fd.UnrollImage(mean2))
 
u1,s1,v1 = np.linalg.svd(CovFaces, full_matrices = True, compute_uv = True)
 
u2,s2,v2 = np.linalg.svd(CovNonFaces, full_matrices = True, compute_uv = True)
 
# fd.PlotSVD(s1)
#  
# fd.PlotSVD(s2)
  
k = 25
 
s1 = np.diag(s1)
s2 = np.diag(s2)
 
u1k = u1[ : , 0 : k]
 
s1k = s1[0 : k, 0 : k]
 
# v1k = v1[0 : k, :]
 
u2k = u2[ : , 0 : k]
 
s2k = s2[0 : k, 0 : k]
 
# v2k = v2[0 : k, :]
 
Faces_Ek = np.dot(np.dot(u1k, s1k), u1k.T)
 
NonFaces_Ek = np.dot(np.dot(u2k, s2k), u2k.T)
 
# iFaces_Ek = np.linalg.inv(Faces_Ek)
# 
# iNonFaces_Ek = np.linalg.inv(NonFaces_Ek)
 
iFaces_Ek = np.dot(np.dot(u1k, np.linalg.inv(s1k)), u1k.T) 
 
iNonFaces_Ek = np.dot(np.dot(u2k, np.linalg.inv(s2k)), u2k.T) 
 
DetFaces_Ek = 1
DetNonFaces_Ek = 1
 
for i in range(k):
    DetFaces_Ek *= s1k[i, i]
    DetNonFaces_Ek *= s2k[i, i]

mu1 = fd.UnrollImage(mean1)
mu2 = fd.UnrollImage(mean2)
 










#   
#      
# result = np.zeros([3, 200])
# correct = np.zeros(3)
# for i in range(200):
#     path = "Testset\\test" + str(i) + ".png"
#     testimage = skimage.img_as_float(skimage.io.imread(path))
#     Input = fd.UnrollImage(testimage)
#     result[0, i] = fd.ComputeGaussianP(Input, mu1, iFaces_Ek, DetFaces_Ek, k)
#     result[1, i] = fd.ComputeGaussianP(Input, mu2, iNonFaces_Ek, DetNonFaces_Ek, k)
#     if i <= 99:
#         if (result[0, i] > result[1, i]):
#             result[2, i] = 1
#             correct[0] += 1
#             correct[2] += 1
#     else:
#         if (result[0, i] < result[1, i]):
#             result[2, i] = 1
#             correct[1] += 1
#             correct[2] += 1
# correct[2] = float(correct[2])/2
# a = 0
 
if (sys.platform == 'linux2'):
    path = "Testset/PhotoCollege.png"
else:
    path = "Testset\\PhotoCollege.png"
    
# copys = 10
#    
#
  
   
if 0:
    tempimage = skimage.img_as_float(skimage.io.imread(path))
    testimage = skimage.transform.rescale(tempimage, 0.5)
       
    skimage.io.imsave(path, testimage)
 
 
if 0:
    detection = fd.DetectFaceG(path, mu1, mu2, iFaces_Ek, DetFaces_Ek, iNonFaces_Ek, DetNonFaces_Ek, k)
    outpath = path.split(".")
    output = outpath[0] + "-result-" + str(i) + ".png"
    skimage.io.imsave(output, detection)


if 0:
    pathes = fd.CreateGPyramid(path, copys, 0.05)
      
    for i in range(copys):
        print i
        detection = fd.DetectFaceG(pathes[i], mu1, mu2, iFaces_Ek, DetFaces_Ek, iNonFaces_Ek, DetNonFaces_Ek, k)
        outpath = path.split(".")
        output = outpath[0] + "-result-" + str(i) + ".png"
        skimage.io.imsave(output, detection)
    a = 0
       
    mean = np.hstack((mean1, mean2))
    pl.imshow(mean)
    pl.show()

 
         









