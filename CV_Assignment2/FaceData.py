import numpy as np
import pylab as pl
import sys
import skimage
import skimage.io
import skimage.transform
import os
import tools as t
import random as rand
from numpy import max


def GenerateFaces(GTFilename):
    CurrentImageName = "null"
    f = open(GTFilename, 'rt')
    for count in range(200):
        line = f.readline()
        if not line:
            break
        linedata = line.split(" ")
        if (CurrentImageName != linedata[0]):
            CurrentImageName = linedata[0]
            if (sys.platform == 'linux2'):
                path = "Trainingset/" + CurrentImageName
            else:
                path = "Trainingset\\" + CurrentImageName
            print path
            tempimage = skimage.img_as_float(skimage.io.imread(path))
            CurrentImage = t.Gradient_to_img(tempimage)               
        if not CurrentImage.any():
            break
        currentcords = np.zeros(12)
        for i in range(12):
            currentcords[i] = float(linedata[i + 1])
        faceimage = t.CropFace(CurrentImage, currentcords, 1.3)
        if count <= 99:
            if (sys.platform == 'linux2'):
                name = "Faces/face" + str(count) + ".png"
            else:
                name = "Faces\\face" + str(count) + ".png"
        else:
            if (sys.platform == 'linux2'):
                name = "Testset/test" + str(count - 100) + ".png"
            else:
                name = "Testset\\test" + str(count - 100) + ".png"
    
        skimage.io.imsave(name, faceimage)
    f.close()
    return

def GenerateNonFaces(size):
    count = -1
    for i in range(10):
        if (sys.platform == 'linux2'):
            path = "NonFaces/" + str(i + 1) + ".jpg"
        else:
            path = "NonFaces\\" + str(i + 1) + ".jpg"
        print path
        tempimage = skimage.img_as_float(skimage.io.imread(path))
        CurrentImage = t.TurnGray(tempimage)
        (height, width, channels) = CurrentImage.shape
        for j in range(10):
            CordX = int(rand.uniform(size, height - size))
            CordY = int(rand.uniform(size, width - size))
            frame = [CordX + size, CordY + size, CordX - size, CordY - size]
            target = t.ResizeFace(t.CropImage(CurrentImage, frame), 12)
            count += 1
            if (sys.platform == 'linux2'):
                name = "NonFaces/nface" + str(count) + ".png"
            else:
#                 name = "NonFaces\\nface" + str(count) + ".png"
                name = "Testset\\test" + str(count + 100) + ".png"
#             name = "Testset\\nface" + str(i * 10 + j) + ".png"
            skimage.io.imsave(name, target)   
    return


def ReadInFaces(Faces, NFaces):
    for index in range(100):
#         if (sys.platform == 'linux2'):
#             path1 = "Faces/face" + str(index) + ".png"
#             path2 = "NonFaces/nface" + str(index) + ".png"
#         else:
#             path1 = "Faces\\face" + str(index) + ".png"
#             path2 = "NonFaces\\nface" + str(index) + ".png"
        path1 = "Testset\\test" + str(index) + ".png"
        path2 = "Testset\\test" + str(index + 100) + ".png"
        Faceimage = skimage.img_as_float(skimage.io.imread(path1))
        NFaceimage = skimage.img_as_float(skimage.io.imread(path2))
        Faces[index] = Faceimage
        NFaces[index] = NFaceimage  
    return
    
    
def UnrollFaces(Faces):#transform a set of image to a m*n matrix, m nD vector, a nD vector for each image
    (num, height, width, channels) = Faces.shape
    Flaten = np.zeros([num, height * width])
    for index in range(num):
        Flaten[index] = UnrollImage(Faces[index])
    return Flaten
    
def UnrollImage(Image):  #transform a image to a 1*n array
    (height, width, channels) = Image.shape
    Unroll = np.zeros(height * width)
    for i in range(height):
            for j in range(width):
                Unroll[i * width + j] = Image[i, j, 0]
    return Unroll
    
    
    
def MakeCombo(Faces, NFaces):
    for i in range(10):
        FaceComboX = Faces[i * 10].copy()
        for j in range(9):
            index = i * 10 + j + 1
            FaceComboX = np.hstack((FaceComboX, Faces[index]))
        if i == 0:
            FaceCombo1 = FaceComboX.copy()
        else:
            FaceCombo1 = np.vstack((FaceCombo1, FaceComboX))
    for i in range(10):
        FaceComboX = NFaces[i * 10].copy()
        for j in range(9):
            index = i * 10 + j + 1
            FaceComboX = np.hstack((FaceComboX, NFaces[index]))
        if i == 0:
            FaceCombo2 = FaceComboX.copy()
        else:
            FaceCombo2 = np.vstack((FaceCombo2, FaceComboX))
    FaceCombo = np.hstack((FaceCombo1, FaceCombo2))
    return FaceCombo
    

def FaceMean(Faces):
    Mean = Faces[0].copy()
    for index in range(1, 100):
        for i in range(12):
            for j in range(12):
                Mean[i, j, 0] += Faces[index, i, j, 0]
    for i in range(12):
            for j in range(12):
                Mean[i, j, 0] /= 100
                Mean[i, j, 1] = Mean[i, j, 0]
                Mean[i, j, 2] = Mean[i, j, 0]
                Mean[i, j, 3] = 1
    return Mean
        
    
def CovMatrix(FacesVec, mean):
    (num, dim) = FacesVec.shape
    TempM = np.zeros([num, dim])
    for i in range(num):
        TempM[i] = FacesVec[i] - mean
    TempMT = TempM.T
    Cov = np.dot(TempMT, TempM)
    return Cov



def PlotSVD(S):
#     means_men = np.zeros(100)
#     for i in range(100):
#         means_men[i] = rand.uniform(1,1000)
#     index = np.arange(100)
    index = np.arange(S.shape[0])
    bar_width = 1  
    opacity = 0.4  
    plot = pl.bar(index, S, bar_width, alpha=opacity, color='b',label= 'singular value')  
    pl.title('Result of SVD')  
    pl.ylim(0,S.max())  
    pl.xlim(0,S.max())  
    pl.tight_layout()  
    pl.show()  



def ComputeGaussianP(Input, mu, iEk, DetSk, k):
#     a = 1/(np.power(2 * np.pi, float(k)/2.0) * np.sqrt(DetSk))
    a = 1
    b = Input - mu
    c = -0.5 * np.dot(np.dot(b, iEk), b.T)
    result = a * np.power(np.e, c)
    return result




def GenerateW(Faces, NonFaces, LR, Itr, diff = 5):
    w = np.zeros(145)
    Sum = np.zeros(145)  
    for n in range(Itr):
        for i in range(100):
            wx = np.dot(w, Faces[i])
            g = 1/(1 + np.power(np.e, -wx))
            Sum += (1 - g) * Faces[i]
            wx = np.dot(w, NonFaces[i])
            g = 1/(1 + np.power(np.e, -wx))
            Sum += -g * NonFaces[i]
        Sum *= LR
        w += Sum
        w_diff = np.linalg.norm(Sum)
        if (w_diff < diff):
            break
    return w




def NMSFaceG(Location, Value, threshold, radius):  # non-maxima suppression for face detection with gaussian.
    #Location is the matrix for face location, Value is the corresponding matrix for the P value
    Suppressed = np.zeros(Location.shape)
    (height, width) = Location.shape
    for i in range(radius, height - radius):
        for j in range(radius, width - radius):
            if Location[i, j]:
                istart = max(i - radius, 0)
                iend = min(i + radius + 1, height)
                jstart = max(j - radius, 0)
                jend = min(j + radius + 1, width)
                positive = 1
                for m in range(istart, iend):
                    for n in range(jstart, jend):
                        if Value[0, i, j] < Value[0, m, n]:
                            positive = 0
                if (positive == 1 and Value[1, i, j] > threshold and Value[0, i, j] >0.85):
                    Suppressed[i, j] = 1
    return Suppressed


def NMSFaceL(Location, Value, threshold, radius):  # non-maxima suppression for face detection with Linear
    #Location is the matrix for face location, Value is the corresponding matrix for the P value
    Suppressed = np.zeros(Location.shape)
    (height, width) = Location.shape
    for i in range(radius, height - radius):
        for j in range(radius, width - radius):
            if Location[i, j] and Value[i, j] > threshold:
                istart = max(i - radius, 0)
                iend = min(i + radius + 1, height)
                jstart = max(j - radius, 0)
                jend = min(j + radius + 1, width)
                positive = 1
                for m in range(istart, iend):
                    for n in range(jstart, jend):
                        if Value[i, j] < Value[m, n]:
                            positive = 0
                            break
                    if positive == 0:
                        break
                if (positive == 1):
                    Suppressed[i, j] = 1
    return Suppressed



def DetectFaceG(path, mu1, mu2, iFaces_Ek, DetFaces_Ek, iNonFaces_Ek, DetNonFaces_Ek, k):
    testimage = skimage.img_as_float(skimage.io.imread(path))
    (height, width, channels) = testimage.shape
    
    Location = np.zeros([height, width])
    Value = np.zeros([2, height, width])
    
    for i in range(height - 11):
        for j in range(width - 11):
            Input = testimage[i : i + 12, j : j + 12, :]
            Input = UnrollImage(Input)
            result1 = ComputeGaussianP(Input, mu1, iFaces_Ek, DetFaces_Ek, k)
            result2 = ComputeGaussianP(Input, mu2, iNonFaces_Ek, DetNonFaces_Ek, k)
            if (result1 > result2):
                Location[i + 6, j + 6] = 1
                Value[0, i + 6, j + 6] = result1
                Value[1, i + 6, j + 6] = result1 - result2
    
    Slocation = NMSFaceG(Location, Value, 0.07, 6) 
    
    for i in range(height - 12):
        for j in range(width - 12):
            if Slocation[i, j]:
                t.Marker(testimage, i, j, 6)

#     pl.imshow(testimage)
#     pl.show()
    return testimage



def DetectFaceL(path, w):
    testimage = skimage.img_as_float(skimage.io.imread(path))
    (height, width, channels) = testimage.shape
    
    Location = np.zeros([height, width])
    Value = np.zeros([height, width])
    
    for i in range(height - 12):
        for j in range(width - 12):
            tempInput = testimage[i : i + 12, j : j + 12, :]
            Input = UnrollImage(tempInput)
            
            InputVec = np.zeros(145)
            InputVec[144] = 1
            
            for k in range(144):
                 InputVec[k] = Input[k]
            wx = np.dot(w, InputVec)
            g = 1/(1 + np.power(np.e, -wx/7))

            if g == 1:
                testnumber = 0
            if (g > 0.5):
                Location[i + 6, j + 6] = 1
                Value[i + 6, j + 6] = g
    
    Slocation = NMSFaceL(Location, Value, 0.9, 10) 
    
    for i in range(height - 12):
        for j in range(width - 12):
            if Slocation[i, j]:
#             if Location[i, j]:
                t.Marker(testimage, i, j, 6)
        
#     pl.imshow(testimage)
#     pl.show()
    return testimage



def CreateGPyramid(path, copys, pace):
    pathes = []
    name = path.split("\\")
    name = name[len(name) - 1]
    name = name.split(".")[0]
    testimage = skimage.img_as_float(skimage.io.imread(path))
    testimage = skimage.transform.rescale(testimage, 0.1)
    for i in range(copys):
        if (sys.platform == 'linux2'):
            outpath = "Testset" + "/" + name + "-" + str(i) + ".png"
        else:
            outpath = "Testset" + "\\" + name + "-" + str(i) + ".png"
#         tempimage = skimage.transform.rescale(testimage, 1 - pace * (i + 1))
#         skimage.io.imsave(outpath, tempimage)
        pathes.append(outpath)
    return pathes