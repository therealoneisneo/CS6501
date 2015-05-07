# tools.py
# Spring 2015 CS_6501 Computer Vision Final
# by Frank

import numpy as np
import copy as c
import sys

import skimage.io





def SaveMatrix(filename, Matrix):
    (height, width) = Matrix.shape
    output = open(filename, 'w')
    output.write("size " + str(height) + " " + str(width) + "\n")
    for i in range(height):
        for j in range(width):
            output.write(str(Matrix[i, j]) + " ")
        output.write("\n")
    output.close()
    return


def SaveMatrix2CSV(filename, Matrix):
    (height, width) = Matrix.shape
    output = open(filename, 'w')
#     output.write("size " + str(height) + " " + str(width) + "\n")
    for i in range(height):
        for j in range(width):
            output.write(str(Matrix[i, j]) + ",")
        output.write("\n")
    output.close()
    return


def ReadDisMatrix(filename):
    input = open(filename, 'r')
    line = input.readline().strip('\n')
    linedata = line.split(" ")
    if (linedata[0] == "size"):
        height = int(linedata[1])
        width = int(linedata[2])
    DisMatrix = np.zeros([height, width])
    for i in range(height):
        line = input.readline().strip('\n')
        linedata = line.split(" ")
        for j in range(width):
            DisMatrix[i, j] = float(linedata[j])
    input.close()
    return DisMatrix



def ReadImages(imagelist): # read image names from "list" file and read 
                    # images and store in "Images" and return Images
    f = open(imagelist, 'rt')
    Images = []
    for i in range(10000):
        line = f.readline()
        if not line:
            break
#         if (i%2 == 0):
#             continue
        ImageName = line.strip('\n')
        if (sys.platform == 'linux2'):
            path = "ImageSet/" + ImageName
        else:
            path = "ImageSet\\" + ImageName
        print path
        CurrentImage = skimage.img_as_float(skimage.io.imread(path))  
#         Images[index] = c.copy(CurrentImage)
        Images.append(c.copy(CurrentImage))
    f.close()
    return Images

def ColorDistanceMatrix(Images):
    print "Computing Color Distance Matrix..."
    size = len(Images)
    DisMatrix = np.zeros([size,size])
    for i in range(size):
        for j in range(i, size):
            print "computing " + str(i) + " to " + str(j) + " in " + str(size) 
            DisMatrix[i, j] = ColorDistance(Images[i], Images[j])
            DisMatrix[j, i] = DisMatrix[i, j]
    return DisMatrix


def EDistanceOfVecs(VecM):
    (num, dim) = VecM.shape
    ans = np.empty([num,num])
    for i in range(num):
        for j in range(i, num):
            temp = np.subtract(VecM[i], VecM[j])
            ans[i, j] = np.linalg.norm(temp)
            ans[j, i] = ans[i, j]
    return ans




def GrayDistanceMatrix(Images):
    print "Computing Gray Distance Matrix..."
    size = len(Images)
    DisMatrix = np.zeros([size,size])
    FlatVec = ImagesToVecArray(Images)
    for i in range(size):
        for j in range(i, size):
            print "computing " + str(i) + " to " + str(j) + " in " + str(size)
#             DisMatrix[i, j] = GrayDistance(Images[i], Images[j])
            DisMatrix[i, j] = np.linalg.norm(np.subtract(FlatVec[i], FlatVec[j]))
            DisMatrix[j, i] = DisMatrix[i, j]
    return DisMatrix



def ColorDistance(Image1, Image2):
    distance = 0.0
    (height, width, channels) = Image1.shape
#     pixelDis = np.zeros([width,channels])
#     wRGB = [1.0, 1.0, 1.0]  # the weight for 3 channels, could be [0.3, 0.59, 0.11]
    wRGB = [0.3, 0.59, 0.11]
    for i in range(height):
        for j in range(width):
            for k in range(3):
                temp = np.power(wRGB[k] * (Image1[i,j,k] - Image2[i,j,k]), 2)
                distance += temp 
    distance = np.sqrt(distance)
    return distance

def GrayDistance(Image1, Image2):
    distance = 0.0
    (height, width, channels) = Image1.shape
    for i in range(height):
        for j in range(width):
            distance += np.power(np.linalg.norm(Image1[i,j] - Image2[i,j]), 2)
    distance = np.sqrt(distance)
    return distance



def outputCords(CordM, filename): # output cords as dim* num
    (height, width) = CordM.shape
    output = open(filename, 'w')
    for j in range(width):
        output.write("v ")
        for i in range(height):
            output.write(str(CordM[i, j]) + " ")
        output.write("\n")
    output.close()
    return





def CreateQMatrix1(DistanceM): # based on the function on section 8.2.1 classical scaling and principal coordinates
    (height, width) = DistanceM.shape  #    version with asymmetrical matrix
    if height != width:
        print "Dimensionality dosen't match!"
        return
    n = height
    DistanceMSq = np.copy(DistanceM)
    Qn = np.zeros([n, n])
    sumDisMSq = 0.0
    
    for i in range(n):
        for j in range(n):
            DistanceMSq[i,j] = np.power(DistanceMSq[i,j], 2)
            sumDisMSq += DistanceMSq[i,j]
            
    sumDisMSq /= (n*n)
    ColumnSum = np.zeros(n)
    
    for i in range(n):
        for j in range(n):
            ColumnSum[i] += DistanceMSq[i, j]
    
    ColumnSum /= float(n)

    for i in range(n):
        for j in range(n):
            Qn[i,j] = -(DistanceMSq[i,j] - 2 * ColumnSum[i] + sumDisMSq)/2
    return Qn


def CreateQMatrix2(DistanceM): # based on the function on section 8.2.1 classical scaling and principal coordinates
    (height, width) = DistanceM.shape  #    version with symmetrical matrix
    if height != width:
        print "Dimensionality dosen't match!"
        return
    n = height
    DistanceMSq = np.copy(DistanceM)
    Qn = np.zeros([n, n])
    sumDisMSq = 0.0
    
    for i in range(n):
        for j in range(n):
            DistanceMSq[i,j] = np.power(DistanceMSq[i,j], 2)
            sumDisMSq += DistanceMSq[i,j]
            
    sumDisMSq /= (n*n)
    
    ColumnSum = np.zeros(n)
    
    for i in range(n):
        for j in range(n):
            ColumnSum[i] += DistanceMSq[i, j]
    
    ColumnSum /= float(n)
    
    for i in range(n):
#         for k in range(n):
#             ColumnSum += DistanceMSq[i,k]
#         ColumnSum *= (float(2)/float(n))
        for j in range(i, n):
            Qn[i,j] = -(DistanceMSq[i,j] - ColumnSum[i] - ColumnSum[j] + sumDisMSq)/2
            Qn[j,i] = Qn[i,j]
    return Qn


def ImageToGrayArray(image):
    (height, width, channels) = image.shape
    ans = np.empty([height, width])
    for i in range(height):
        for j in range(width):
            ans[i, j] = (image[i, j, 0] + image[i, j, 1] + image[i, j, 2])/3
    return ans
                
        


def ImagesToVecArray(Images): # flaten images in to 1D vectors
    n = len(Images)
    (height, width) = (Images[0].shape[0],Images[0].shape[1])
    VecLength = height * width
#     graymatrix = np.empty([n, height, width])
#     for i in range(n):
#         graymatrix = 
    ans = np.empty([n, VecLength])
    for i in range(n):
        print "Flatening..." + str(i*100/n) + "%"
        temp = ImageToGrayArray(Images[i])
#         for j in range (height)
        ans[i] = c.copy(temp.flatten())
    return ans









