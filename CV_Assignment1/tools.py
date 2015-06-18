import numpy as np
import copy as c
import sys
import skimage.io
import skimage.transform


DoGThreshold = 0.015


def BuildGKernel(sigma, kernelsize, Dimension): # build Gaussian kernel, Dimension take in 1 or 2 only
    PI = np.pi
    center = int(kernelsize/2)
    if Dimension == 1:
        kernel = np.zeros(kernelsize)       # create a 1D kernel for the Gsmooth
       
        for i in range(center + 1): #build the kernel
            dist = abs(center - i)# the distance to the center of the kernel
            kernel[i] = np.exp( - dist / (2 * sigma * sigma)) / (np.sqrt(2 * PI) * sigma)
            kernel[kernelsize - 1 - i] = kernel[i]
    elif Dimension == 2:
        kernel = np.zeros([kernelsize,kernelsize])       # create a 2D kernel for the Gsmooth
        for i in range(center + 1): #build the kernel
            for j in range(center + 1):
                dist = ((center - i)**2 + (center - j)**2)   # the distance to the center of the kernel
                kernel[i, j] = np.exp( - dist / (2 * sigma * sigma)) / (2 * PI * sigma * sigma)
                kernel[kernelsize - 1 - i, kernelsize - 1 - j] = kernel[i, j]
                kernel[i, kernelsize - 1 - j] = kernel[i, j]
                kernel[kernelsize - 1 - i, j] = kernel[i, j]
    else:
        print "error: Dimension parameter for Gaussian kernel must be 1 or 2!"
    return kernel





def Gsmooth(I, sigma = 0.8, kernelsize = 3.0, channels = 3):        # Gaussian Smooth
    # I is the input image, sigma is the parameter for the Gsmooth, kernelsize is the size of the convolution mask
    # channels can indicate the image is a gray matrix or a color image
    if channels > 1:
        (height, width, channels) = I.shape
    else:
        (height, width) = I.shape
    
    if kernelsize >= np.minimum(height, width):
        return I
    center = int(kernelsize/2)
    kernel = BuildGKernel(sigma, kernelsize, 1)
#     kernel = np.zeros(kernelsize)       # create a 1D kernel for the Gsmooth
#     center = int(kernelsize/2)
#     PI = 3.1415926
#     for i in range(center + 1): #build the kernel
#         dist = abs(center - i)# the distance to the center of the kernel
#         kernel[i] = kernel[kernelsize - 1 - i] = np.exp( - dist / (2 * sigma * sigma)) / (np.sqrt(2 * PI) * sigma)
#    ans = np.zeros(I.shape)
#    ans1 = np.zeros(I.shape)
    ans = c.copy(I)

    sums = width + height
    
    weight = 0.0
    for i in range(kernelsize):
        weight += kernel[i]
    
    if channels > 1:
        # first round, smooth in x direction
        percent = 0.0
        for x in range(height):
            if ((x * 100) / sums) != percent:
                percent = ((x * 100) / sums)
                print " Smoothing...", percent, "% complete"
            for y in range(center, width - center + 1):
                for z in range(channels):
                    if z == 3:
                        ans[x, y, z] = 1
                    else:
                        ans[x, y ,z] = 0
                        for i in range(kernelsize):
                            ans[x, y, z] += kernel[i] * I[x, y - center + i, z]
                        ans[x, y, z] /= weight
                            
#                        for i in range(center, width - center + 1):
#                            
#                        if y < center:
#                            for i in range(-y, center + 1):
#                                ans[x,y,z] += kernel[center + i] * I[x, y + i, z]
#                        elif y > (width - 1 - center):
#                            for i in range(-center, width - y):
#                                ans[x,y,z] += kernel[center + i] * I[x, y + i, z]
#                        else:
#                            for i in range(-center, center + 1):
#                                ans[x,y,z] += kernel[center + i] * I[x, y + i, z]
#                            
                            

        # second round, smooth in y direction
        ans1 = c.copy(ans)
        for y in range(width):
            if (((y + height) * 100) / sums) != percent:
                percent = (((y + height) * 100) / sums)
                print " Smoothing...", percent, "% complete"
            for x in range(center, height - center + 1):
                for z in range(channels):
                    if z == 3:
                        ans1[x, y, z] = 1
                    else:
                        ans1[x, y ,z] = 0
                        for i in range(kernelsize):
                            ans1[x, y, z] += kernel[i] * ans[x- center + i, y, z]
                        ans1[x, y, z] /= weight   
    
    else:
        # first round, smooth in x direction
        percent = 0.0
        for x in range(height):
            if ((x * 100) / sums) != percent:
                percent = ((x * 100) / sums)
                print " Smoothing...", percent, "% complete"
            for y in range(center, width - center):
                ans[x, y] = 0
                for i in range(kernelsize):
                    ans[x, y] += kernel[i] * I[x, y - center + i]
                ans[x, y] /= weight
        # second round, smooth in y direction
        ans1 = c.copy(ans)
        for y in range(width):
            if (((y + height) * 100) / sums) != percent:
                percent = (((y + height) * 100) / sums)
                print " Smoothing...", percent, "% complete"
            for x in range(center, height - center):
                ans1[x, y] = 0
                for i in range(kernelsize):
                    ans1[x, y] += kernel[i] * ans[x- center + i, y]
                ans1[x, y] /= weight   
    return ans1


def Gsmooth2D(I, sigma = 0.8, kernelsize = 3.0):        # Gaussian Smooth 2D version, slower
    # I is the input image, sigma is the parameter for the Gsmooth, kernelsize is the size of the convolution mask

    center = int(kernelsize/2)

#     for i in range(center + 1): #build the kernel
#         for j in range(center + 1):
#             dist = ((center - i)**2 + (center - j)**2)   # the distance to the center of the kernel
#             kernel[i, j] = kernel[kernelsize - 1 - i, kernelsize - 1 - j] = \
#             np.exp( - dist / (2 * sigma * sigma)) / (2 * PI * sigma * sigma)
    kernel = BuildGKernel(sigma, kernelsize, 2)
    weight = 0.0
    for i in range(kernelsize):
        for j in range(kernelsize):
            weight += kernel[i,j]
         
    ans = np.empty(I.shape)
    (height, width, channels) = I.shape
     
    for x in range(height):
        print " Smoothing...", (x * 100) / height, "% complete"
        for y in range(width):
            for c in range(channels - 1):
                for i in range(-center, center + 1):
                    for j in range(-center, center + 1):
                        if (((x + i) > (height - 1)) or ((x + i) < 0) or \
                            ((y + j) > (width - 1)) or ((y + j) < 0)):
                            ans[x,y,c] += kernel[center + i, center + j] * 0
                        else:
                            ans[x,y,c] += kernel[center + i, center + j] * I[x + i, y + j, c]
                ans[x,y,c] /= weight
    for x in range(height):
        for y in range(width):          
            ans[x,y,3] = 1.0
    print "smooth complete"
    return ans


def GenerateCovarianceEig(CM, GX, GY, masksize = 5): #CM.shape :[height,width,2,2] Generate the Coverence Matrix and Eigen value
    height = CM.shape[0]
    width = CM.shape[1]
    halfsize = int(np.floor(masksize/2))
    kernel = Gkernel(1.2, masksize)
    for i in range(halfsize, height - halfsize):
        print "Generating CovMatrix...", (i * 100) / height, "% complete"
        for j in range(halfsize, width - halfsize):
            for m in range(masksize):
                for n in range(masksize):
                    xindex = i - halfsize + m
                    yindex = j - halfsize + n
                    CM[i, j, 0, 0] += GX[xindex, yindex] * GX[xindex, yindex] * kernel[m, n]
                    CM[i, j, 0, 1] += GX[xindex, yindex] * GY[xindex, yindex] * kernel[m, n]
                    CM[i, j, 1, 0] = CM[i, j, 0, 1]
                    CM[i, j, 1, 1] += GY[xindex, yindex] * GY[xindex, yindex] * kernel[m, n]
    EigenM = np.zeros([height, width, 2])
    #TempM = np.zeros([2,2])
    for i in range(halfsize, height - halfsize):
        print "Generating EigValue...", (i * 100) / height, "% complete"
        for j in range(halfsize, width - halfsize):
            EigenM[i,j] = np.linalg.eig(CM[i,j])[0]
    return EigenM


def TruncEigenvalue(EigenM, ThresholdRatio):# sort all lower eigen values and set threshold for it
    height = EigenM.shape[0]
    width = EigenM.shape[1]
    listlenth = height * width
    Pixellist = np.zeros([listlenth,3])
    #EVlist = np.zeros(listlenth)
    
    for i in range(height):
        for j in range(width):
            l1 = EigenM[i, j, 0]
            l2 = EigenM[i, j, 1]
            #Pixellist[i * width + j, 0] = np.minimum(EigenM[i, j, 0], EigenM[i, j, 1])
            #EVlist[i * j + j] = np.minimum(EigenM[i, j])
            Pixellist[i * width + j, 0] = (l1*l2) - 0.04 * (l1 + l2) * (l1 + l2)
            Pixellist[i * width + j, 1] = i
            Pixellist[i * width + j, 2] = j
    
    #Pixellist is a n*3 array, in with each row is a pixel's lower eigen value, x-cord and y-core
            
    #SortedEV = np.sort(EVlist)        
    Threshold = int(np.floor(listlenth * ThresholdRatio))# Threshold is the number of pixels will be discarded in the corner detection
    sequence = np.argsort(Pixellist, axis=0)#sequence is the sequence of sorted Pixellist index 
    ChosenPoints = np.zeros([1,2])
    Temp = np.zeros([1,2])
    ChosenPoints[0, 0] = Pixellist[sequence[listlenth - 1, 0], 1]
    ChosenPoints[0, 1] = Pixellist[sequence[listlenth - 1, 0], 2]
    
    for i in range(listlenth - 2, Threshold, -1):
        Temp[0, 0] = Pixellist[sequence[i, 0], 1]
        Temp[0, 1] = Pixellist[sequence[i, 0], 2]
        ChosenPoints = np.append(ChosenPoints, Temp, axis=0)
    return ChosenPoints

def GenerateCornerPoint(PC, height, width, kernel = 5): # Select the corner points, PC is the list of [eigen, x, y] of Candidate point
    ans = np.zeros([1,2])# ans is just the x,y cord of chosen pixels
    flag = np.zeros([height, width])
    ans[0, 0] = PC[0, 0]
    ans[0, 1] = PC[0, 1]
    flag[ans[0, 0], ans[0, 1]] = 1 
    for i in range(1, len(PC)):
        x = PC[i, 0]
        y = PC[i, 1]
        if flag[x, y] == 0:
            SetMValue(flag, x, y, kernel, 1)
            ans = np.append(ans, [(x, y)], axis=0)
    return ans

def SetMValue(M, x, y, kernelsize, value): # set values in a region in M at location (x,y) with 'kernelsize' as 'value' 
    (height, width) = M.shape
    half = int(np.floor(kernelsize/2))
    xstart = int(np.maximum(x - half, 0))
    ystart = int(np.maximum(y - half, 0))
    xend = int(np.minimum(x + half + 1, height))
    yend = int(np.minimum(y + half + 1, width))
    for i in range(xstart, xend):
        for j in range(ystart, yend):
            M[i, j] = value
    return
    

def CornerPlot(Corners, Image):# draw indicator of corners in Image
    ans = c.copy(Image)
#     (height, width, channels) = Image.shape
    for i in range(len(Corners)):
        x = Corners[i, 0]
        y = Corners[i, 1]
        Marker(ans, x, y, 5)
#         if (x >= 0) and (x + 6< height) and (y >= 0) and (y + 6 < width):
#             for j in range(7):
#                 ans[x + j, y, 0] = 1
#                 ans[x + j, y, 1] = 1
#                 ans[x + j, y, 2] = 0
#                 ans[x + j, y + 6, 0] = 1
#                 ans[x + j, y + 6, 1] = 1
#                 ans[x + j, y + 6, 2] = 0
#                 ans[x , y + j, 0] = 1
#                 ans[x , y + j, 1] = 1
#                 ans[x , y + j, 2] = 0
#                 ans[x + 6, y + j, 0] = 1
#                 ans[x + 6, y + j, 1] = 1
#                 ans[x + 6, y + j, 2] = 0
    return ans


def GrayToArray(I): # get a singel channel of a gray image to an array
    (height, width, channels) = I.shape
    ans = np.empty([height, width])
    percent = 0.0
    for i in range(height):
        temp = (i * 100 / (height - 1))
        if temp != percent:
            percent = temp
            print "Turn to Grayscale..." , percent, "% complete"
        for j in range(width):
            ans[i, j] = I[i, j, 0]
    return ans

def TurnGray(I): # turn the image to gray scale
    (height, width, channels) = I.shape
    ans = np.empty([height, width])
    percent = 0.0
    for i in range(height):
        temp = (i * 100 / (height - 1))
        if temp != percent:
            percent = temp
            print "Turn to Grayscale..." , percent, "% complete"
        for j in range(width):
            ans[i, j] = 0.3 * I[i, j, 0] + 0.59 * I[i, j, 1] + 0.11 * I[i, j, 2]
    return ans



def ComputeXorYGradient_Pixel(I, x, y, Gtype): # Compute the Gradient of a pixel in a image. Gtype can be: 0: GX 1: GY 2: GXX 3:GYY 4:GXY 5:All
    tempI = np.zeros([5, 5])
    for i in range(5):
        for j in range(5):
            tempI[i, j] = I[x - 2 + i, y - 2 + j]
    Gx = ComputeXorYGradient(tempI, 0)
    Gy = ComputeXorYGradient(tempI, 1)
    tempGx = np.zeros([3, 3])
    tempGy = np.zeros([3, 3])
    for i in range(3):
        for j in range(3):
            tempGx[i, j] = Gx[1 + i, 1 + j]
            tempGy[i, j] = Gy[1 + i, 1 + j]
    Gxx = ComputeXorYGradient(tempGx, 0)
    Gyy = ComputeXorYGradient(tempGy, 1)
    Gxy = ComputeXorYGradient(tempGx, 1)
    if Gtype == 0:
        return tempGx[1, 1]
    elif Gtype == 1:
        return tempGx[1, 1]
    elif Gtype == 2:
        return Gxx[1, 1]
    elif Gtype == 3:
        return Gyy[1, 1]
    elif Gtype == 4:
        return Gxy[1, 1]
    else:
        return (tempGx[1, 1], tempGy[1, 1], Gxx[1, 1], Gyy[1, 1], Gxy[1, 1])



def ComputeXorYGradient(I, Gtype): # I is the input, Gtype: 0: x direction, 1: y direction
    
    height = I.shape[0]
    width = I.shape[1]     
    ans = np.empty([height, width])
    sq2 = np.sqrt(2)
    SobelX = np.array([[-1,-sq2,-1],[0,0,0],[1,sq2,1]])
    SobelY = np.array([[-1,0,1],[-sq2,0,sq2],[-1,0,1]])
#     ans1 = np.empty(I.shape)
#     ans2 = np.empty(I.shape)
#     ans = np.empty(I.shape)
    for i in range(height):
        ans[i, 0] = 0
        ans[i, width - 1] = 0
    for j in range(width):
        ans[0, j] = 0
        ans[height - 1, j] = 0
    #-----------------------------------------------------------
    if Gtype == 0:
        #compute the gradient on x direction
        if len(I.shape) > 2:    
            k = 0
            for i in range(1, height - 1):
#                print"Computing X Gradient...",  (i * 100) / (height - 1), "% complete..."
                for j in range(1, width - 1):
                    ans[i,j] = I[i - 1, j - 1, k] * SobelX[0,0] + \
                           I[i - 1, j, k] * SobelX[0,1] + \
                           I[i - 1, j + 1, k] * SobelX[0,2] + \
                           I[i, j - 1, k] * SobelX[1,0] + \
                           I[i, j, k] * SobelX[1,1] + \
                           I[i, j + 1, k] * SobelX[1,2] + \
                           I[i + 1, j - 1, k] * SobelX[2,0] + \
                           I[i + 1, j, k] * SobelX[2,1] + \
                           I[i + 1, j + 1, k] * SobelX[2,2]
        else:
            for i in range(1, height - 1):
#                print"Computing X Gradient...",  (i * 100) / (height - 1), "% complete..."
                for j in range(1, width - 1):
                    ans[i,j] = I[i - 1, j - 1] * SobelX[0,0] + \
                           I[i - 1, j] * SobelX[0,1] + \
                           I[i - 1, j + 1] * SobelX[0,2] + \
                           I[i, j - 1] * SobelX[1,0] + \
                           I[i, j] * SobelX[1,1] + \
                           I[i, j + 1] * SobelX[1,2] + \
                           I[i + 1, j - 1] * SobelX[2,0] + \
                           I[i + 1, j] * SobelX[2,1] + \
                           I[i + 1, j + 1] * SobelX[2,2]
    else:
        #compute the gradient on y direction
        if len(I.shape) > 2:    
            k = 0
            for i in range(1, height - 1):
#                print"Computing Y Gradient...",  (i * 100) / (height - 1), "% complete..."
                for j in range(1, width - 1):
                    ans[i,j] = I[i - 1, j - 1, k] * SobelY[0,0] + \
                           I[i - 1, j, k] * SobelY[0,1] + \
                           I[i - 1, j + 1, k] * SobelY[0,2] + \
                           I[i, j - 1, k] * SobelY[1,0] + \
                           I[i, j, k] * SobelY[1,1] + \
                           I[i, j + 1, k] * SobelY[1,2] + \
                           I[i + 1, j - 1, k] * SobelY[2,0] + \
                           I[i + 1, j, k] * SobelY[2,1] + \
                           I[i + 1, j + 1, k] * SobelY[2,2]
        else:
            for i in range(1, height - 1):
#                print"Computing Y Gradient...",  (i * 100) / (height - 1), "% complete..."
                for j in range(1, width - 1):
                    ans[i,j] = I[i - 1, j - 1] * SobelY[0,0] + \
                           I[i - 1, j] * SobelY[0,1] + \
                           I[i - 1, j + 1] * SobelY[0,2] + \
                           I[i, j - 1] * SobelY[1,0] + \
                           I[i, j] * SobelY[1,1] + \
                           I[i, j + 1] * SobelY[1,2] + \
                           I[i + 1, j - 1] * SobelY[2,0] + \
                           I[i + 1, j] * SobelY[2,1] + \
                           I[i + 1, j + 1] * SobelY[2,2]
    return ans

def ComputeGradient(I, GX, GY): #compute the gradient of x and y and stored in GX and GY
    (height, width) = I.shape     
#    sq2 = np.sqrt(2)
#    SobelX = np.array([[-1,-sq2,-1],[0,0,0],[1,sq2,1]])
#    SobelY = np.array([[-1,0,1],[-sq2,0,sq2],[-1,0,1]])
#     ans1 = np.empty(I.shape)
#     ans2 = np.empty(I.shape)
#     ans = np.empty(I.shape)
    #-----------------------------------------------------------
    # frame
    for i in range(height):
        GX[i, 0] = 0
        GX[i, width - 1] = 0
    for j in range(width):
        GY[0, j] = 0
        GY[height - 1, j] = 0
    #----------------------------------------------------------
    # convolution
    GX = ComputeXorYGradient(I, 0)
    GY = ComputeXorYGradient(I, 1)
    return 


def Sobel(I,GradM): # Sobel kernel for gradient: I is the input image(smoothed) and GradM is the matrix stores the direction of gradient
    PI = 3.1415926
    (height, width, channels) = I.shape     
    sq2 = np.sqrt(2)
    SobelX = np.array([[-1,-sq2,-1],[0,0,0],[1,sq2,1]])
    SobelY = np.array([[-1,0,1],[-sq2,0,sq2],[-1,0,1]])
    ans1 = np.empty(I.shape)
    ans2 = np.empty(I.shape)
    ans = np.empty(I.shape)
    #-----------------------------------------------------------
    # frame
    for i in range(height):
        for k in range(channels):
            #ans[i, 0, k] = I[i, 0, k]
            #ans[i, width - 1, k] = I[i, width - 1, k]
            ans[i, 0, k] = 0
            ans[i, width - 1, k] = 0
    for j in range(width):
        for k in range(channels):
            #ans[0, j, k] = I[0, j, k]
            #ans[height - 1, j, k] = I[height - 1, j, k]
            ans[0, j, k] = 0
            ans[height - 1, j, k] = 0
    #----------------------------------------------------------
    # convolution
    for i in range(1, height - 1):
        print"Computing Gradient...",  (i * 100) / (height - 1), "% complete..."
        for j in range(1, width - 1):
                for k in range(channels - 1):
                    ans1[i,j,k] = I[i - 1, j - 1, k] * SobelX[0,0] + \
                           I[i - 1, j, k] * SobelX[0,1] + \
                           I[i - 1, j + 1, k] * SobelX[0,2] + \
                           I[i, j - 1, k] * SobelX[1,0] + \
                           I[i, j, k] * SobelX[1,1] + \
                           I[i, j + 1, k] * SobelX[1,2] + \
                           I[i + 1, j - 1, k] * SobelX[2,0] + \
                           I[i + 1, j, k] * SobelX[2,1] + \
                           I[i + 1, j + 1, k] * SobelX[2,2]
                           
                    ans2[i,j,k] = I[i - 1, j - 1, k] * SobelY[0,0] + \
                           I[i - 1, j, k] * SobelY[0,1] + \
                           I[i - 1, j + 1, k] * SobelY[0,2] + \
                           I[i, j - 1, k] * SobelY[1,0] + \
                           I[i, j, k] * SobelY[1,1] + \
                           I[i, j + 1, k] * SobelY[1,2] + \
                           I[i + 1, j - 1, k] * SobelY[2,0] + \
                           I[i + 1, j, k] * SobelY[2,1] + \
                           I[i + 1, j + 1, k] * SobelY[2,2]
    #------------------------------------------------------------
    # compute gradient
    for i in range(1, height - 1):
        for j in range(1, width - 1):
                for k in range(3):
                    ans[i, j, k] = np.sqrt(ans1[i, j, k]**2 + ans2[i, j, k]**2)
                    if ans1[i, j, k] == 0:
                        GradM[i, j] += PI/2
                    else:
                        GradM[i, j] += np.arctan(ans2[i, j, k] / ans1[i, j, k])
                GradM[i, j] /= 3    
    if channels > 3:  
        for i in range(height):
            for j in range(width):
                ans[i, j, 3] = 1.0
            
            
#     for i in range(height):
#         for j in range(width):
#             for k in range(1, channels - 1):
#                 ans[i, j, 0] += ans[i, j, k]
#             ans[i, j, 1] = ans[i, j, 0]
#             ans[i, j, 2] = ans[i, j, 0]
    return ans




def NMS(I, G):  # non-maxima suppression. I is the input gradient image and G is the direction matrix
    (height, width, channels) = I.shape
    ans = c.copy(I)
    PI = 3.1415926
    DirectM = np.empty(G.shape)
    direction = 0 #  0:+- 22.5, 1: 22.5 ~ 67.5, 2: -22.5 ~ -67.5 , 3:rest 
    D = np.zeros(5)
    print "Calculating direction..."
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            D[0] = abs(G[i, j])            
            D[1] = abs(G[i, j] - PI / 4)
            D[2] = abs(G[i, j] + PI / 4)
            D[3] = abs(G[i, j] - PI / 2)
            D[4] = abs(G[i, j] + PI / 2)
            Min = 10000
            for k in range(5):
                if D[k] < Min :
                    Min = D[k]
                    if k >= 3:
                        direction = 3
                    else:
                        direction = k
            DirectM[i, j] = direction
    for i in range(1, height - 1):
        print"suppresing...",  (i * 100) / (height - 1), "% complete..."
        for j in range(1, width - 1):
            if I[i,j,0] != 0 :
                if DirectM[i, j] == 0:
                    if (I[i, j, 0] < I[i, j + 1, 0]) or (I[i, j, 0] < I[i, j - 1, 0]):
                        ans[i, j, 0] = 0
                        ans[i, j, 1] = 0
                        ans[i, j, 2] = 0 
                elif DirectM[i, j] == 1:
                    if (I[i, j, 0] < I[i - 1, j + 1, 0]) or (I[i, j, 0] < I[i + 1, j - 1, 0]):
                        ans[i, j, 0] = 0
                        ans[i, j, 1] = 0
                        ans[i, j, 2] = 0
                elif DirectM[i, j] == 2:
                    if (I[i, j, 0] < I[i + 1, j, 0]) or (I[i, j, 0] < I[i - 1, j, 0]):
                        ans[i, j, 0] = 0
                        ans[i, j, 1] = 0
                        ans[i, j, 2] = 0
                else:
                    if (I[i, j, 0] < I[i + 1, j + 1, 0]) or (I[i, j, 0] < I[i - 1, j - 1, 0]):
                        ans[i, j, 0] = 0
                        ans[i, j, 1] = 0
                        ans[i, j, 2] = 0
    return ans
    
    
    
def Diff(I1, I2): #compute the Diff of two images. the result is I2 - I1 
    (height, width) = I1.shape
    ans = np.empty(I1.shape)
    for i in range(height):
        for j in range(width):
            ans[i, j] = I2[i, j] - I1[i, j]
#     ans = ScaleMapping(ans, 0, 1.0)
#     ans = ValueShift(ans)
    return ans

def ADiff(I1, I2): # the abs diff of two images
    (height, width, channels) = I1.shape
    ans = np.empty(I1.shape)
    for i in range(height):
        for j in range(width):
            for k in range(channels):
                if k == 3:
                    ans[i, j, k] = 1.0
                else:
                    ans[i, j, k] = abs(I1[i, j ,k] - I2[i, j ,k])
    return ans
    
    
    
def Matrix_to_img(G):# transform a matrix to an image for visulization
    (a,b) = G.shape
    ans = np.zeros([a,b,4])
    for i in range(a):
        for j in range(b):
            ans[i,j,0] = ans[i,j,1] = ans[i,j,2] = G[i,j]
            ans[i,j,3] = 1.0
    return ans
            

def ValueShift(I, BaseValue = 0):
    minv = 10000
    (height, width, channels) = I.shape
    ans = c.copy(I)
    for i in range(height):
        for j in range(width):
            if I[i,j,0] < minv:
                minv = I[i,j,0]
    for i in range(height):
        for j in range(width):
            ans[i,j,0] = I[i,j,0] - minv + BaseValue
            ans[i,j,1] = ans[i,j,0]
            ans[i,j,2] = ans[i,j,0] 
    return ans

            
def ScaleMapping(I, scalemin = 0, scalemax = 1.0): # mapping a grayscale image to range scalemin~scalemax
    maxv = -10000
    minv = 10000
    (height, width) = I.shape
    ans = c.copy(I)
    for i in range(height):
        for j in range(width):
            if I[i, j] > maxv:
                maxv = I[i, j]
            elif I[i, j] < minv:
                minv = I[i, j]
    D = maxv - minv
    Srange = scalemax - scalemin
    for i in range(height):
        for j in range(width):
            ans[i, j] = (I[i, j] - minv) * Srange / D + scalemin
    return ans
    
    
def Double_Threshold(I,H_ratio):# Hysteresis thresholding the edge
    L_ratio = 0.5
    Histo = np.zeros(256)
    mapped = ScaleMapping(I, 255)
    (height, width, channels) = I.shape
    for i in range(height):
        for j in range(width):
            if I[i,j,0] > 0:
                Histo[int(mapped[i,j,0])] += 1
    count = 0
    bucket = np.zeros(256)
    for i in range(256):
        count += Histo[i]
        bucket[i] = count
    H_count = count * H_ratio
    L_count = L_ratio * H_count
    hset = lset = 0
    for i in range(255):
        if (bucket[i] < L_count) and (bucket[i + 1] >= L_count):
            L_Threshold = i;
            lset = 1
        if (bucket[i] < H_count) and (bucket[i + 1] >= H_count):
            H_Threshold = i;
            hset = 1
            break
    ans = np.empty(I.shape)
    
    
    for i in range(1, height - 1):
        print"Thresholding...",  (i * 100) / (height - 1), "% complete..."
        for j in range(1, width - 1):
            if(mapped[i,j,0] > H_Threshold):
                ans[i,j,0] = 1
                ans[i,j,1] = ans[i,j,0]
                ans[i,j,2] = ans[i,j,0]
                if lset:   
                    TrackEdeg(I, ans, i, j, L_Threshold) 
            if channels > 3:
                ans[i,j,3] = 1.0
    return ans

def TrackEdeg(I, O, ii, jj, L_threshold):# sub-routine for the edge tracking
    Xindex = [0, -1, -1, -1, 0, 1, 1, 1]
    Yindex = [1, 1, 0, -1, -1, -1, 0, 1]
    for k in range(8):
        i = ii + Xindex[k]
        j = jj + Yindex[k]
        if (I[i,j,0] > L_threshold) and (O[i,j,0] == 0):
            O[i,j,0] = 1
            TrackEdeg(I, O, i, j, L_threshold)
    return
    
    
    
def GenerateGaussianLayers(I, Gnum):# I is the input image, Lnum is the number of Layers
    print "Generating Gaussian Layers..."
    (height, width) = I.shape
    ans = np.zeros([Gnum, height, width])
    sigma = np.zeros(Gnum)#the sigma values for different guassian
    s = Gnum - 2
#     k = np.sqrt(2)
    k = np.power(2, (1.0/s))
    sigmazero = 1.6
    for i in range(Gnum - 1):
        sigma[i] = sigmazero * np.power(k,i)
    ans[0] = I
    for i in range(Gnum - 1):
        ans[i + 1] = Gsmooth(I, sigma[i], 7, 1)
#        ans[i + 1] = ans[i + 1][4:-4, 4:-4]
#    ans[0] = ans[0][4:-4, 4:-4]
    return ans
        
    
def GenerateScales(I, Snum):# I is the input image, Snum is the number of different scales
    #when generating the scale pyramid, first create a double sized I as the -1 layer 
    
    Scales = {0 : ScaleImage(I, 2)}
    for i in range(1, Snum):
        Scales[i] =  ScaleImage(Scales[i - 1])
    return Scales


def ScaleImage1(I, scale = 1):#sample an image to a new scale by factor 2
                                        # the 'scale' only take 1 as down-scale and 2 as up scale
                                        #only take in grayscale image
    (height, width, channels) = I.shape
    if scale == 1:
        height = int(np.floor(height/2))
        width = int(np.floor(width/2))
        ans = np.zeros([height, width, channels])
        if channels > 3:
            for i in range(height):
                for j in range(width):
                    ans[i, j, 3] = 1
        for i in range(height):
            for j in range(width):
                ans[i, j, 0] += I[i * 2, j * 2, 0]
                ans[i, j, 0] += I[i * 2 + 1, j * 2, 0] 
                ans[i, j, 0] += I[i * 2, j * 2 + 1, 0] 
                ans[i, j, 0] += I[i * 2 + 1, j * 2 + 1, 0]
                ans[i, j, 0] /= 4
                ans[i, j, 1] = ans[i, j, 0]
                ans[i, j, 2] = ans[i, j, 0]
    else:
        height = height * 2
        width = width * 2
        ans = np.zeros([height, width, channels])
        if channels > 3:
            for i in range(height):
                for j in range(width):
                    ans[i, j, 3] = 1
        for i in range(height):
            for j in range(width):
                ans[i, j, 0] = I[int(np.floor(i / 2)), int(np.floor(j / 2)), 0]
                ans[i, j, 1] = ans[i, j, 0]
                ans[i, j, 2] = ans[i, j, 0]
    return ans

def ScaleImage2(I, scale = 1):#simply up-sample or down sample to a new scale by factor 2. the 'scale' only take 1 as down-scale and 2 as up scale. only take in 1 channel image(matrix)
    (height, width) = I.shape
    if scale == 1: # down-scale
        height = int(np.floor(height/2))
        width = int(np.floor(width/2))
        ans = np.zeros([height, width])
        for i in range(height):
            for j in range(width):
                ans[i, j] = I[i * 2, j * 2]
    else:
        height = height * 2
        width = width * 2
        ans = np.zeros([height, width])
        for i in range(height):
            for j in range(width):
                ans[i, j] = I[int(np.floor(i / 2)), int(np.floor(j / 2))]
    return ans



def ScaleImage(I, scale = 1):#simply up-sample or down sample to a new scale by factor 2
                                        # the 'scale' only take 1 as down-scale and 2 as up scale
                                        #only take in grayscale image
    (height, width, channels) = I.shape
    if scale == 1:
        height = int(np.floor(height/2))
        width = int(np.floor(width/2))
        ans = np.zeros([height, width, channels])
        if channels > 3:
            for i in range(height):
                for j in range(width):
                    ans[i, j, 3] = 1
        for i in range(height):
            for j in range(width):
                ans[i, j, 0] = I[i * 2, j * 2, 0]
                ans[i, j, 1] = ans[i, j, 0]
                ans[i, j, 2] = ans[i, j, 0]
    else:
        height = height * 2
        width = width * 2
        ans = np.zeros([height, width, channels])
        if channels > 3:
            for i in range(height):
                for j in range(width):
                    ans[i, j, 3] = 1
        for i in range(height):
            for j in range(width):
                ans[i, j, 0] = I[int(np.floor(i / 2)), int(np.floor(j / 2)), 0]
                ans[i, j, 1] = ans[i, j, 0]
                ans[i, j, 2] = ans[i, j, 0]
    return ans

def DiffofGaussian(GP, Snum, Gnum): # GP is the GaussianPyramid
    DGnum = Gnum - 1
# c.copy(Gnum)
#    DGnum -=1
    DoG = {(0,0):0}
    for i in range(Snum):
        for j in range(DGnum):
            print "DiffofGaussian processing " + str(i) + " " + str(j)
#            DoG[(i,j)] = Diff(GP[(i, j)], GP[(i, j + 1)])
            DoG[(i,j)] = GP[(i, j)] - GP[(i, j + 1)]
    return DoG


def DoGPyramidDisplay(DoG, Snum, Gnum): #shift the value in a DoG Pyramid for display
    DoGDis = c.copy(DoG)
    for i in range(Snum):
        (height, width) = DoG[i, 0].shape
        for j in range(Gnum - 1):
#            ScaleMapping(DoGDis[i, j], 0.0, 1.0)
            DoGDis[i, j] = DoG[i, j] + 0.2
    return DoGDis


    
def ExtractDoGExtrema(DoG, Snum, DoGnum): # compare each pixel value of DoG with the 26 neigboring pixels
    ans = c.copy(DoG)
    v = np.zeros(27)
    percent = -1.0 #debug use
    sums = Snum * (DoGnum - 1)
    for i in range(Snum):
        for j in range(1, DoGnum - 1):
            if ((i * j * 100) / sums) != percent:
                percent = ((i * j * 100) / sums)
                print "Extracting Extrema...", percent, "% complete"
            (height, width) = ans[i, j].shape
            ans[i, j] = np.zeros(ans[i, j].shape)
            for x in range(10, height - 10):
                for y in range(10, width - 10):
                    v[0] = DoG[i, j][x, y]
                    v[1] = DoG[i, j][x - 1, y - 1]
                    v[2] = DoG[i, j][x - 1, y]
                    v[3] = DoG[i, j][x - 1, y + 1]
                    v[4] = DoG[i, j][x, y - 1]
                    v[5] = DoG[i, j][x, y + 1]
                    v[6] = DoG[i, j][x + 1, y - 1]
                    v[7] = DoG[i, j][x + 1, y]
                    v[8] = DoG[i, j][x + 1, y + 1]
                    v[9] = DoG[i, j - 1][x, y]
                    v[10] = DoG[i, j - 1][x - 1, y - 1]
                    v[11] = DoG[i, j - 1][x - 1, y]
                    v[12] = DoG[i, j - 1][x - 1, y + 1]
                    v[13] = DoG[i, j - 1][x, y - 1]
                    v[14] = DoG[i, j - 1][x, y + 1]
                    v[15] = DoG[i, j - 1][x + 1, y - 1]
                    v[16] = DoG[i, j - 1][x + 1, y]
                    v[17] = DoG[i, j - 1][x + 1, y + 1]
                    v[18] = DoG[i, j + 1][x, y,]
                    v[19] = DoG[i, j + 1][x - 1, y - 1]
                    v[20] = DoG[i, j + 1][x - 1, y]
                    v[21] = DoG[i, j + 1][x - 1, y + 1]
                    v[22] = DoG[i, j + 1][x, y - 1]
                    v[23] = DoG[i, j + 1][x, y + 1]
                    v[24] = DoG[i, j + 1][x + 1, y - 1]
                    v[25] = DoG[i, j + 1][x + 1, y]
                    v[26] = DoG[i, j + 1][x + 1, y + 1]
                    
                    maxcheck = 0
                    mincheck = 0
                    equalcheck = 0
                    if abs(v[0]) >= DoGThreshold:# this is a threshold to be determined by the input image
                        for k in range (1, 27):
                            if v[k] == v[0]:
                                equalcheck += 1
                                break
                            if v[k] > v[0]:
                                mincheck += 1
                            if v[k] < v[0]:
                                maxcheck += 1
                            if maxcheck > 0 and mincheck > 0:
                                break
                        if maxcheck == 26 or mincheck == 26:
                            ans[i, j][x, y] = 1     
    ans1 = {(0,0):0}
    for i in range(Snum):
        for j in range(1, DoGnum - 1):
            ans1[i, j - 1] = ans[i, j]
    return ans1




def filepath(rawpath):# adapt the file path to the system
    parsing = rawpath.split('/')
    n = len(parsing)
    ans = ""
    if (sys.platform == 'linux2'):
        ans = rawpath
    else:
        for i in range(n - 1):
            ans += parsing[i]
            ans += "\\"            
        ans += parsing[n - 1]
    return ans





def Marker(I, x, y, size = 4):# Mark the position (x,y) in I with a (2*size + 1) * (2*size + 1) frame in yellow
    (height, width, channels) = I.shape
    if(np.minimum(height, width) < (size * 2 + 1)):
        return
    if (x + size) >= height:
        x = height - size - 1
    if (y + size) >= width:
        y = width - size - 1
    if (x < size):
        x = size
    if (y < size):
        y = size
    for i in range(size + 1):
        for k in range(3):
            if k == 2:
                I[x + size, y + i, k] = 0
                I[x + size, y - i, k] = 0
                I[x - size, y + i, k] = 0
                I[x - size, y - i, k] = 0
                I[x + i, y + size, k] = 0
                I[x + i, y - size, k] = 0
                I[x - i, y + size, k] = 0
                I[x - i, y - size, k] = 0
            else:
                I[x + size, y + i, k] = 1
                I[x + size, y - i, k] = 1
                I[x - size, y + i, k] = 1
                I[x - size, y - i, k] = 1
                I[x + i, y + size, k] = 1
                I[x + i, y - size, k] = 1
                I[x - i, y + size, k] = 1
                I[x - i, y - size, k] = 1
    return
    

def ValueScale(M, lower, upper):
    if len(M.shape) <= 2:
        (height, width) = M.shape
        IM = c.copy(M)
        vmax = -1000000000
        vmin = 1000000000
        for i in range(height):
            for j in range(width):
                if M[i, j] > vmax:
                    vmax = float(M[i, j])
                if M[i, j] < vmin:
                    vmin = float(M[i, j])
        valuerange = vmax - vmin
        scalerange = upper - lower
        for i in range(height):
            for j in range(width):
                IM[i, j] -= vmin
                IM[i, j] /= valuerange
                IM[i, j] *= scalerange
                IM[i ,j] += lower
        return IM            
    else:
            (height, width, channels) = M.shape
            IM = c.copy(M)
            vmax = -1000000000
            vmin = 1000000000
            for i in range(height):
                for j in range(width):
                    if M[i, j, 0] > vmax:
                        vmax = float(M[i, j, 0])
                    if M[i, j, 0] < vmin:
                        vmin = float(M[i, j, 0])
            valuerange = vmax - vmin
            scalerange = upper - lower
            for i in range(height):
                for j in range(width):
                    IM[i, j, 0] -= vmin
                    IM[i, j, 0] /= valuerange
                    IM[i, j, 0] *= scalerange
                    IM[i ,j, 0] += lower
                    IM[i ,j, 1] = IM[i ,j, 0]
                    IM[i ,j, 2] = IM[i ,j, 0]
                    if channels > 3:
                        IM[i ,j, 3] = 1                        
            return IM                        
            
           
def ReadInPyramid(ImageName, Itype, Snum, Layernum):
    Name = ImageName.split('.')
    ans = {(0,0):0} #initialize a dictionary
    for i in range(Snum):
        for j in range(Layernum):
            temppath = Name[0] + Itype + str(i) + str(j) + ".jpg" 
            print "Reading " + temppath
            ans[i, j] = skimage.img_as_float(skimage.io.imread(temppath))
#             ans[i, j] = TurnGray(ans[i, j])
            ans[i, j] = ans[i, j][:, :, 0]
    return ans
            
            
#             
#              fname = t.filepath("testcases/buildingDoG%d%d.jpg"  % (i, j))
#         skimage.io.imsave(fname, DoGPyramid[(i,j)])
#             
            
            
def SavePyramid(ImageName, Pyramid, Itype, Snum, Gnum):
    Name = ImageName.split('.')
    for i in range(Snum):
        for j in range(Gnum):
            temppath = Name[0] + Itype + str(i) + str(j) + ".jpg" 
            print "Saving " + temppath
            skimage.io.imsave(temppath, Matrix_to_img(Pyramid[(i,j)]))
    return


def KeypointsFilter(IG, Extremas, DoG, rim = 10, ratio = 10, threshold = 0.02): #Extremas is the stack of info of all extrema points.
    for k in range(len(Extremas)):
        extrema = Extremas[k]
        if extrema[0] != -1:
             scale = extrema[0]
             layer = extrema[1] + 1 # correspond to DoG layer
             c = extrema[2]
             l = extrema[3]
             Base = DoG[scale, layer]    
             (height, width) = Base.shape
             if (c < rim) or (c > (height - rim)) or (l < rim) or (l > (width - rim)):
                 extrema[0] = -1
                 break
             
             x = int(float(c) / height * IG.shape[0])
             y = int(float(l) / width * IG.shape[1])
             gradient = ComputeXorYGradient_Pixel(IG, x, y, 5)
             Dxx = gradient[2]
             Dyy = gradient[3]
             Dxy = gradient[4]
             Tr = Dxx + Dyy
             Det = Dxx * Dyy - Dxy * Dxy
             if Det == 0:
                 ratioValue = 1000
             else:
                 ratioValue = (Tr) * (Tr) / Det
             if (np.abs(DoG[scale, layer][c, l]) < threshold) or (ratioValue > ratio):
                 extrema[0] = -1
    return       
            
#def KeypointsFilter(IG, Extrema, DoG, rim = 10, ratio = 10, threshold = 0.02): #Extremas is the info of a single extrema point. 
#    (height, width) = Extrema.shape
##    Gradient = {0:0}
##    for i in range(5):
##        Gradient[i] = c.copy(DoG)
#    #Gradient: 0: x, 1: y, 2:xx, 3:yy, 4:xy
##    Gradient[0] = ComputeXorYGradient(DoG, 0)
##    Gradient[1] = ComputeXorYGradient(DoG, 1)
##    Gradient[2] = ComputeXorYGradient(Gradient[0], 0)
##    Gradient[3] = ComputeXorYGradient(Gradient[1], 1)
##    Gradient[4] = ComputeXorYGradient(Gradient[0], 1)
#    for i in range(height):
#        for j in range(width):
#            if (i < rim) or (i > (height - rim)) or (j < rim) or (j > (width - rim)):
#                Extrema[i, j] = 0
#            
#            elif Extrema[i, j] > 0:
##                Dxx = Gradient[2][i, j]
##                Dyy = Gradient[3][i, j]
##                Dxy = Gradient[4][i, j]
#                x = int(float(i) / height * IG.shape[0])
#                y = int(float(j) / width * IG.shape[1])
#                gradient = ComputeXorYGradient_Pixel(IG, x, y, 5)
#                Dxx = gradient[2]
#                Dyy = gradient[3]
#                Dxy = gradient[4]
#                Tr = Dxx + Dyy
#                Det = Dxx * Dyy - Dxy * Dxy
#                if Det == 0:
#                    ratioValue = 1000
#                else:
#                    ratioValue = (Tr) * (Tr) / Det
#                if (np.abs(DoG[i, j]) < threshold) or (ratioValue > ratio):
#                    Extrema[i, j] = 0
#    return
       
       
       
       
#def RefineExtrema(IG, Extremas, DoG, Snum, Gnum, rim = 10, ratio = 10, threshold = 0.02):
#    extremas = c.copy(Extremas)
#    percent = 0.0
#    sums = Snum * (Gnum - 2)
#    for i in range(Snum):
#        for j in range(1, Gnum - 2):
#            if ((i * (j - 1) * 100) / sums) != percent:
#                percent = ((i * (j - 1) * 100) / sums)
#                print "Refining Extrema...", percent, "% complete"
#            KeypointsFilter(IG, extremas[i, j - 1], DoG[i, j], rim, ratio, threshold)
#    return extremas

def RefineExtrema(IG, Extremas, DoG, Snum, Gnum, rim = 10, ratio = 10, threshold = 0.02): # to re-adjust the location of keypoints to subpixel, Extremas is a stack of extrema points info
    extremas = c.copy(Extremas)
    AdjustKeypoints(extremas, DoG)
    KeypointsFilter(IG, extremas, DoG, rim, ratio, threshold)
    temp = {0:0}
    count = 0
    for i in range(len(extremas)):
        if extremas[i][0] >= 0:
            temp[count] = c.copy(extremas[i])
            count += 1
    extremas = temp
            
    return extremas



def AdjustKeypoints(Extremas, DoG): # adjust a "Extremas" points location, or discard it
    image_scale = 1.0 / 255
    deriv_scale = 0.5 * image_scale
    deriv_2_scale = image_scale
    cross_scale = 0.25 * image_scale
    for j in range(len(Extremas)):
        extrema = Extremas[j]
        count = 0    
        for i in range(5):
            scale = int(extrema[0])
            layer = int(extrema[1]) + 1 # correspond to DoG layer
            if layer == 4:
                a = 0
            c = extrema[2]
            l = extrema[3]
            Base = DoG[scale, layer]
            Prev = DoG[scale, layer - 1]
            Next = DoG[scale, layer + 1] 
            dx = (Base[c, l + 1] - Base[c, l - 1]) * deriv_scale
            dy = (Base[c + 1, l] - Base[c - 1, l]) * deriv_scale
            ds = (Next[c, l] - Prev[c, l]) * deriv_scale
            d1 = [dx, dy, ds]
            
            center = Base[c, l] * 2
            dxx = (Base[c, l + 1] + Base[c, l - 1] - center) * deriv_2_scale
            dyy = (Base[c + 1, l] + Base[c - 1, l] - center) * deriv_2_scale
            dss = (Next[c, l] + Prev[c, l] - center) * deriv_2_scale
            dxy = (Base[c + 1, l + 1] - Base[c + 1, l - 1] - Base[c - 1, l + 1] + Base[c - 1, l - 1]) * cross_scale
            dxs = (Next[c, l + 1] - Next[c, l - 1] - Prev[c, l + 1] + Prev[c, l - 1]) * cross_scale
            dys = (Next[c + 1, l] - Next[c - 1, l] - Prev[c + 1, l] + Prev[c - 1, l]) * cross_scale
    
            Hessian = [[dxx, dxy, dxs],[dxy, dyy, dys],[dxs, dys, dss]]
            result = np.linalg.solve(Hessian, d1) * -1
            if np.abs(result[0]) < 0.5 or np.abs(result[1]) < 0.5 or np.abs(result[2]) < 0.5:
                break
            extrema[2] += np.ceil(result[0])
            extrema[3] += np.ceil(result[1])
            extrema[1] += np.ceil(result[2])
            (height, width) = Base.shape
            if (extrema[1] < 1 or extrema[1] >= 3) or (extrema[2] < 0 or extrema[2] > height) or (extrema[3] < 0 or extrema[3] > width):
                extrema[0] = -1    # marked as in valid
                break
            count += 1
        if count == 5:
            extrema[0] = -1    # marked as in valid       
    return
 
def SaveExStack(ExStack, filename):
    f = open(filename, 'w')
    n = len(ExStack)
    for i in range(n):
        for j in range(4):
            f.write(str(int(ExStack[i][j])))
            f.write(" ")
        f.write("\n")
    f.close()
    return
    
    
def ReadExStack(filename):
    ExStack = {0:0}
    f = open(filename, 'r')
    count = 0
    for line in f:
        line.strip('\n')
        templine = line.split(" ")
        temp = [int(templine[0]),int(templine[1]),int(templine[2]),int(templine[3])]
        ExStack[count] = temp
        count += 1
    f.close()
    return ExStack
    
def ExtremaLocations(Extremas, Snum, Layernum): #gather up the locations in the full size image of extremas in all scales. Only for Display use. the discriptor should be generated on each scale to maintain its scale invarient features
    temp = np.zeros(4)
    ans = {0:0}
    count = 0
#    for i in range(1):
#        for j in range(1):#debug
    for i in range(Snum):
        for j in range(Layernum):
            (height, width) = Extremas[i, j].shape
            for m in range(height):
                for n in range(width):
                    if Extremas[i, j][m, n] == 1:
                        print count
#                        x = float(m) / float(height)
#                        y = float(n) / float(width)
                        temp[0] = i
                        temp[1] = j
                        #temp[2] = x
                        #temp[3] = y  # x, y is for the scaled location, m,n is for the actual location in its corresponding scale
                        temp[2] = m
                        temp[3] = n
                        ans[count] = c.copy(temp)
                        count += 1
    return ans



#def PixelMagDir(Image, x, y):# return the magnitude and direction of a pixel's gradient
#    a = Image[x + 1, y]
#    b = Image[x - 1, y]
#    c = Image[x, y + 1]
#    d = Image[x, y - 1]
#    X = a - b
#    Y = c - d
#    mag = np.sqrt((a - b) * (a - b) + (c - d) * (c - d))
#    theta = np.arctan2(Y, X)  * 180 / np.pi
#    return (mag, theta)
    
def PixelMagDir(Image, x, y):# return the magnitude and direction of a pixel's gradient. Based on Sobel
    a = 2 * Image[x + 1, y] + Image[x + 1, y + 1] + Image[x + 1, y - 1]
    b = 2 * Image[x - 1, y] + Image[x - 1, y + 1] + Image[x - 1, y - 1]
    c = 2 * Image[x, y + 1] + Image[x + 1, y + 1] + Image[x - 1, y + 1]
    d = 2 * Image[x, y - 1] + Image[x + 1, y - 1] + Image[x - 1, y - 1]
    GY = a - b
    GX = c - d
    mag = np.sqrt(GX * GX + GY * GY)
    theta = np.arctan2(GY, GX)  * 180 / np.pi
    return (mag, theta)


def SelectBin(direc, binNum):
    offset = 360.0 / binNum    
    lower = 0.0
    upper = lower + offset
    if direc < 0:
        direc = 360 + direc
    if direc == 360:
            direc = 0
    for i in range(binNum):
        if (direc >= lower) and (direc < upper):
            return i
        else:
            lower += offset
            upper += offset
    return -1

def CreateBins(neigborMagDir, binNum):
    bins = np.zeros(binNum) # bins for "binNum" of directions
    n = neigborMagDir.shape[0]
    # or n = len(neigborMagDir)
    for i in range(n):
        direc = neigborMagDir[i, 4]
        binindex = SelectBin(direc, binNum)
        if binindex == -1:
            print "Bin Select Error!"
            return -1
        else:
            bins[binindex] += float(neigborMagDir[i, 3])
    return bins


def GetKeyDirection(Base, Extrema):
    (height, width) = Base.shape
#    x = int(height * Extrema[2])
#    y = int(width * Extrema[3])
    x = Extrema[2]
    y = Extrema[3]
    offset = range(-4, 5)
    offset.remove(0)
#    positive = range(1, 5)
#    negative = range(-4, 0).reverse()
    neigborMagDir = np.zeros([64, 5]) # cord-x and cord-y, kernel weight, Mag and Dir
    kernel = BuildGKernel(2.0, 9, 2) # the sigma of the kernel still need to be determined
    count = 0
    for i in range(8):
        for j in range(8):
            neigborMagDir[count, 0] = x + offset[i]
            neigborMagDir[count, 1] = y + offset[j]
            neigborMagDir[count, 2] = kernel[4 + offset[i], 4 + offset[j]] * 1000
            count += 1
    for i in range(64):
        xx = neigborMagDir[i, 0]
        yy = neigborMagDir[i, 1]
        (mag, theta) = PixelMagDir(Base, xx, yy)
        neigborMagDir[i, 3] = mag * neigborMagDir[i, 2]
        neigborMagDir[i, 4] = theta

    binNum = 36
    bins = CreateBins(neigborMagDir, binNum)
    binmax = -1.0 # bin max magnitude
    binauxmax = -1.0 # bin aux max magnitude
    binmaxi = -1 # bin max magnitude index
    binauxmaxi = -1 # bin aux max magnitude index
    for i in range(binNum):
        if bins[i] > binmax:
            binmax = bins[i]
            binauxmaxi = binmaxi
            binmaxi = i
        elif bins[i] > binauxmax:
            binauxmaxi = i
    if binauxmax < binmax * 0.8:
        binauxmaxi = -1
    # currently does not return the "binauxmaxi" as a secondary main direction
    
    return (binmaxi, binNum, binmax) 


def RotateImagebyMDir(Image, Maindir, Center): # the rotation routine during the creation of descriptor
    dirIndex = Maindir[0]
    binNum = Maindir[1]
    offset = 360.0 / binNum
    direction = offset * (float(dirIndex) + 0.5) 
    RImage = skimage.transform.rotate(Image, direction, False, Center)
    return RImage


def GrayImageMarker(I, x , y, color = 0): # mark a cross on a gray image, 1 is white and 0 is black
    
   (height, width) = I.shape
   for i in range(height):
       I[i, y] = color
   for i in range(width):
       I[x, i] = color
   return


def BuildDescriptor(GP, Extrema, BinNum): #build a sift descriptor for a key point "Extrema"
    Scale = Extrema[0]
    Layer = Extrema[1] + 1 # +1 for shift?
    Base = GP[Scale, Layer]
    (height, width) = Base.shape
#    x = int(height * Extrema[2])
#    y = int(width * Extrema[3])
    x = Extrema[2]
    y = Extrema[3]
    
    mainDirection = GetKeyDirection(Base, Extrema)
    
    RotatedI = RotateImagebyMDir(Base, mainDirection, (y, x)) # the skimage.transform rotate take in the rotation center as (y, x) rather than (x, y)
    
#    GrayImageMarker(RotatedI, x, y, 0)
#   
#    skimage.io.imsave("siftdata/rotated.jpg", RotatedI)  #debug
    
    #rotate the image by its mainDirection
    ind = range(-8, 9)
    ind.remove(0)
    neigborMagDir = np.zeros([256, 5]) # cord-x and cord-y, kernel weight, Mag and Dir
    kernel = BuildGKernel(3.5, 17, 2) * 1000 # the sigma of the kernel still need to be determined
    count = 0
    for i in range(16):
        for j in range(16):
            neigborMagDir[count, 0] = x + ind[i]
            neigborMagDir[count, 1] = y + ind[j]
            neigborMagDir[count, 2] = kernel[8 + ind[i], 8 + ind[j]]
            count += 1
    for i in range(256):
        xx = neigborMagDir[i, 0]
        yy = neigborMagDir[i, 1]
        (mag, theta) = PixelMagDir(RotatedI, xx, yy)
        neigborMagDir[i, 3] = mag * neigborMagDir[i, 2]
        neigborMagDir[i, 4] = theta
    #------------build bins and Gkernel weight BuildGKernel(sigma, kernelsize, Dimension)
    
    
    ExVec = GenerateExVector(neigborMagDir)

    return ExVec 

def GenerateExVector(neigborMagDir):
    vec = np.zeros(128)
    for i in range(4):
        for j in range(4):
            ii = i * 4
            jj = j * 4
            for m in range(4):
                for n in range(4):
                    absx = ii + m
                    absy = jj + n
                    neigbor_pos = absx * 16 + absy
                    vec_seg_pos = (i * 4 + j) * 8
                    bin_index = SelectBin(neigborMagDir[neigbor_pos, 4], 8)
                    vec_pos = vec_seg_pos + bin_index
                    vec[vec_pos] += neigborMagDir[neigbor_pos, 3]
    return vec
                
                 
def GenerateDescriptors(ExStack, GP): #main routine to generate 128 dimensional vector descriptors for all extrema points of the image
#    (Snum, Gnum) = (GP.shape[0], GP.shape[1])
    ExNum = len(ExStack)
    descriptors = {0:0}
    for i in range(ExNum):
        print "Computing the" + str(i) + "of" + str(ExNum) + "feature points..."
        descriptors[i] = BuildDescriptor(GP, ExStack[i], 8)
    #build a collection of descriptors here
    return descriptors
            
def ArrowMarker(I, x1, y1, x2, y2): # mark an arrow on the image
    k = float(y2 - y1) / float(x2 - x1)
    height = I.shape[0]
    width = I.shape[1]
    if x1 < x2:
        for x in range(x1, x2):
            
            y = k * (x - x1) + y1
            if (y - np.floor(y)) < 0.5:
                y = np.floor(y)
            else:
                y = np.ceil(y)
            if x >= height or y >= width:
                break
            I[x, y, 0] = 1.0
            I[x, y, 1] = 1.0
            I[x, y, 2] = 0.0
    else:
        for x in range(x2, x1):
            y = k * (x - x1) + y1
            if (y - np.floor(y)) < 0.5:
                y = np.floor(y)
            else:
                y = np.ceil(y)
            if x >= height or y >= width:
                break
            I[x, y, 0] = 1.0
            I[x, y, 1] = 1.0
            I[x, y, 2] = 0.0
            
            
    k =  float(x2 - x1) / float(y2 - y1)
    if y1 < y2:
        for y in range(y1, y2):
            x = k * (y - y1) + x1
            if (x - np.floor(x)) < 0.5:
                x = np.floor(x)
            else:
                x = np.ceil(x)
            if x >= height or y >= width:
                break
            I[x, y, 0] = 1.0
            I[x, y, 1] = 1.0
            I[x, y, 2] = 0.0
    else:
        for y in range(y2, y1):
            x = k * (y - y1) + x1
            if (x - np.floor(x)) < 0.5:
                x = np.floor(x)
            else:
                x = np.ceil(x)
            if x >= height or y >= width:
                break
            I[x, y, 0] = 1.0
            I[x, y, 1] = 1.0
            I[x, y, 2] = 0.0
    if x2 < height and y2 < width:
        I[x2, y2, 0] = 1.0
        I[x2, y2, 1] = 1.0
        I[x2, y2, 2] = 0.0
    Marker(I, x1, y1, 2)
    return

            
def DisplayKeyPoints(ExStackDict, GP, IM):
    I = c.copy(IM)
    PI = np.pi
    (height, width, channels) = I.shape
    for i in range(len(ExStackDict)):
        ExStack = ExStackDict[i]
        Scale = ExStack[0]
        Layer = ExStack[1] + 1 # +1 for shift?
        Base = GP[Scale, Layer]
        (height1, width1) = Base.shape
        x = ExStack[2]
        y = ExStack[3]
        temp = float(x)/height1       
        x = temp
        temp = float(y)/width1
        y = temp
        x = int(x * height)
        y = int(y * width)
        MDir = GetKeyDirection(Base, ExStack)
        dirIndex = MDir[0]
        binNum = MDir[1]
        Magnitude = MDir[2]
        offset = 360.0 / binNum
        direction = offset * (float(dirIndex) + 0.5)
        reg = direction * 2 * PI / 360

        Xoffset = int(Magnitude * np.cos(reg))
        Yoffset = int(Magnitude * np.sin(reg))   
        if Xoffset == 0:
            Xoffset = 1
        if Yoffset == 0:
            Yoffset = 1
        ArrowMarker(I, x, y, x + Xoffset, y + Yoffset)
    return I
        
        
            