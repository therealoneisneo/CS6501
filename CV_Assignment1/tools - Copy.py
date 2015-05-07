import numpy as np
import copy as c

def blur(I):                              # Naive blur
    ans = np.empty(I.shape)
    (h, w, channels) = I.shape
    for y in range(h):
        for x in range(w):
            y1 = max(y-1,0)
            x1 = max(x-1,0)
            y3 = min(y+1,h-1)
            x3 = min(x+1,w-1)
            for c in range(channels):
                ans[y,x,c] = (I[y1,x1,c]+I[y1,x,c]+I[y1,x3,c]+
                              I[y,x1,c]+I[y,x,c]+I[y,x3,c]+
                              I[y3,x1,c]+I[y3,x,c]+I[y3,x3,c])/9
    return ans


def Gsmooth(I, sigma = 0.8, kernelsize = 3.0):        # Gaussian Smooth
    # I is the input image, sigma is the parameter for the Gsmooth, kernelsize is the size of the convolution mask
    kernel = np.zeros(kernelsize)       # create a 1D kernel for the Gsmooth
    center = int(kernelsize/2)
    PI = 3.1415926
    for i in range(center + 1): #build the kernel
        dist = abs(center - i)# the distance to the center of the kernel
        kernel[i] = kernel[kernelsize - 1 - i] = np.exp( - dist / (2 * sigma * sigma)) / (np.sqrt(2 * PI) * sigma)
        
    compute_weight = False
    weight = 0.0
    for i in range(len(kernel)):
        weight += kernel[i]
        
    ans = np.empty(I.shape)
    ans1 = np.empty(I.shape)
    (height, width, channels) = I.shape
    sums = width + height
    # first round, smooth in x direction
    for x in range(height):
        print " Smoothing...", (x * 100) / sums, "% complete"
        for y in range(width):
            for c in range(channels):
                if y < center:
                    weight = 0.0
                    diff = center - y
                    for i in range(diff,kernelsize):
                        weight += kernel[i]
                    compute_weight = True
                    for i in range(-center + diff, center + 1):
                        ans[x,y,c] += kernel[center + i] * I[x, y + i, c]
                    ans[x,y,c] /= weight
                elif y > (width - 1 - center):
                    weight = 0.0
                    diff = y - (width - 1 - center)
                    for i in range(kernelsize - diff):
                        weight += kernel[i]
                    compute_weight = True
                    for i in range(-center, center + 1 - diff):
                        ans[x,y,c] += kernel[center + i] * I[x, y + i, c]
                    ans[x,y,c] /= weight
                else:
                    if compute_weight:
                        weight = 0.0
                        for i in range(len(kernel)):
                            weight += kernel[i]
                        compute_weight = False
                    for i in range(-center, center + 1):
                        ans[x,y,c] += kernel[center + i] * I[x, y + i, c]
                    ans[x,y,c] /= weight
    # second round, smooth in y direction
    for y in range(width):
        print " Smoothing...", ((y + height) * 100) / sums, "% complete"
        for x in range(height):
            for c in range(channels):
                if x < center:
                    weight = 0.0
                    diff = center - x
                    for i in range(diff,kernelsize):
                        weight += kernel[i]
                    compute_weight = True
                    for i in range(-center + diff, center + 1):
                        ans1[x,y,c] += kernel[center + i] * ans[x + i, y, c]
                    ans1[x,y,c] /= weight
                elif x > (width - 1 - center):
                    weight = 0.0
                    diff = x - (width - 1 - center)
                    for i in range(kernelsize - diff):
                        weight += kernel[i]
                        compute_weight = True
                    for i in range(-center, center + 1 - diff):
                        ans1[x,y,c] += kernel[center + i] * ans[x + i, y, c]
                    ans1[x,y,c] /= weight
                else:
                    if compute_weight:
                        weight = 0.0
                        for i in range(len(kernel)):
                            weight += kernel[i]
                        compute_weight = False
                    for i in range(-center, center + 1):
                        ans1[x,y,c] += kernel[center + i] * ans[x + i, y, c]
                    ans1[x,y,c] /= weight
    return ans1


def Gsmooth2D(I, sigma = 0.8, kernelsize = 3.0):        # Gaussian Smooth 2D
    # I is the input image, sigma is the parameter for the Gsmooth, kernelsize is the size of the convolution mask
    kernel = np.zeros([kernelsize,kernelsize])       # create a 2D kernel for the Gsmooth
    center = int(kernelsize/2)
    PI = 3.1415926
    for i in range(center + 1): #build the kernel
        for j in range(center + 1):
            dist = ((center - i)**2 + (center - j)**2)   # the distance to the center of the kernel
            kernel[i, j] = kernel[kernelsize - 1 - i, kernelsize - 1 - j] = \
            np.exp( - dist / (2 * sigma * sigma)) / (2 * PI * sigma * sigma)
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





def Gkernel(sigma = 0.8, kernelsize = 5):        # generate Gaussian weight kernel 
    kernel = np.zeros([kernelsize,kernelsize])       # create a 2D kernel for the Gsmooth
    center = int(kernelsize/2)
    PI = 3.1415926
    for i in range(center + 1): #build the kernel
        for j in range(center + 1):
            dist = ((center - i)**2 + (center - j)**2)   # the distance to the center of the kernel
            kernel[i, j] = kernel[kernelsize - 1 - i, kernelsize - 1 - j] = \
            kernel[i, kernelsize - 1 - j] =\
            kernel[kernelsize - 1 - i, j] =\
            np.exp( - dist / (2 * sigma * sigma)) / (2 * PI * sigma * sigma)
#     weight = 0.0
#     for i in range(kernelsize):
#         for j in range(kernelsize):
#             weight += kernel[i,j]
    return kernel


def GenerateCoverenceEig(CM, GX, GY, masksize = 5): #CM.shape :[height,width,2,2]
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


def TruncEigenvalue(EigenM, ThresholdRatio):
    height = EigenM.shape[0]
    width = EigenM.shape[1]
    listlenth = height * width
    Pixellist = np.zeros([listlenth,3])
    #EVlist = np.zeros(listlenth)
    
    for i in range(height):
        for j in range(width):
            Pixellist[i * width + j, 0] = np.minimum(EigenM[i, j, 0], EigenM[i, j, 1])
            #EVlist[i * j + j] = np.minimum(EigenM[i, j])
            Pixellist[i * width + j, 1] = i
            Pixellist[i * width + j, 2] = j
    
    #Pixellist is a n*3 array, in with each row is a pixel's lower eigen value, x-cord and y-core
            
    #SortedEV = np.sort(EVlist)        
    Threshold = int(np.floor(listlenth * ThresholdRatio))# Threshold is the number of pixels will be discarded in the corner detection
    sequence = np.argsort(Pixellist, axis=0)#sequence is the sequence of sorted Pixellist index 
    ChosenPoints = np.zeros([1,3])
    Temp = np.zeros([1,3])
    ChosenPoints[0] = Pixellist[sequence[listlenth - 1, 0]]
    for i in range(listlenth - 2, Threshold, -1):
        Temp[0] = Pixellist[sequence[i,0]]
        ChosenPoints = np.append(ChosenPoints, Temp, axis=0)
    return ChosenPoints

def GenerateCornerPoint(PC, EM): # Select the corner points, PC is the list of [eigen, x, y] of points, EM is the Eigen value Matrix 
    
    
    
    
    
    
    ans = np.zeros([1,3])
    ans[0] = PC[0]
    for i in range (1, len(PC)):
        if PC[i, 0]:
            ans[i] = PC[i]
    return ans


def CornerPlot(Corners, Image):
    ans = c.copy(Image)
    (height, width, channels) = Image.shape
    for i in range(len(Corners)):
        x = Corners[i, 1] - 3
        y = Corners[i, 2] - 3
        if (x >= 0) and (x + 6< height) and (y >= 0) and (y + 6 < width):
            for j in range(7):
                ans[x + j, y, 0] = 1
                ans[x + j, y, 1] = 1
                ans[x + j, y, 2] = 0
                ans[x + j, y + 6, 0] = 1
                ans[x + j, y + 6, 1] = 1
                ans[x + j, y + 6, 2] = 0
                ans[x , y + j, 0] = 1
                ans[x , y + j, 1] = 1
                ans[x , y + j, 2] = 0
                ans[x + 6, y + j, 0] = 1
                ans[x + 6, y + j, 1] = 1
                ans[x + 6, y + j, 2] = 0
    return ans




def TurnGray(I): # turn the image to gray scale
    (height, width, channels) = I.shape
    ans = np.empty(I.shape)
    for i in range(height):
        print "Turn to Grayscale..." , i * 100 / (height - 1), "% complete"
        for j in range(width):
            ans[i,j,0] = 0.3 * I[i,j,0] + 0.59 * I[i,j,1] + 0.11 * I[i,j,2]
            ans[i,j,1] = ans[i,j,0]
            ans[i,j,2] = ans[i,j,0]
            if channels > 2:
                ans[i,j,3] = 1.0
    return ans


def ComputeGradient(I, GX, GY): #compute the gradient of x and y and stored in GX and GY
    (height, width, channels) = I.shape     
    sq2 = np.sqrt(2)
    SobelX = np.array([[-1,-sq2,-1],[0,0,0],[1,sq2,1]])
    SobelY = np.array([[-1,0,1],[-sq2,0,sq2],[-1,0,1]])
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
    k = 0
    for i in range(1, height - 1):
        print"Computing Gradient...",  (i * 100) / (height - 1), "% complete..."
        for j in range(1, width - 1):
            GX[i,j] = I[i - 1, j - 1, k] * SobelX[0,0] + \
                   I[i - 1, j, k] * SobelX[0,1] + \
                   I[i - 1, j + 1, k] * SobelX[0,2] + \
                   I[i, j - 1, k] * SobelX[1,0] + \
                   I[i, j, k] * SobelX[1,1] + \
                   I[i, j + 1, k] * SobelX[1,2] + \
                   I[i + 1, j - 1, k] * SobelX[2,0] + \
                   I[i + 1, j, k] * SobelX[2,1] + \
                   I[i + 1, j + 1, k] * SobelX[2,2]
                   
            GY[i,j] = I[i - 1, j - 1, k] * SobelY[0,0] + \
                   I[i - 1, j, k] * SobelY[0,1] + \
                   I[i - 1, j + 1, k] * SobelY[0,2] + \
                   I[i, j - 1, k] * SobelY[1,0] + \
                   I[i, j, k] * SobelY[1,1] + \
                   I[i, j + 1, k] * SobelY[1,2] + \
                   I[i + 1, j - 1, k] * SobelY[2,0] + \
                   I[i + 1, j, k] * SobelY[2,1] + \
                   I[i + 1, j + 1, k] * SobelY[2,2]
    return 


def Sobel(I,GradM): # Sobel kernel for gradien: I is the input image(smoothed) and GradM is the matrix stores the direction of gradient
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
                for k in range(channels - 1):
                    ans[i, j, k] = np.sqrt(ans1[i, j, k]**2 + ans2[i, j, k]**2)
                    if ans1[i, j, k] == 0:
                        GradM[i, j] += PI/2
                    else:
                        GradM[i, j] += np.arctan(ans2[i, j, k] / ans1[i, j, k])
                GradM[i, j] /= 3      
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
    
    
    
def Diff(I1, I2):
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
    
    
    
def Gradient_to_img(G):
    (a,b) = G.shape
    ans = np.zeros([a,b,4])
    for i in range(a):
        for j in range(b):
            ans[i,j,0] = ans[i,j,1] = ans[i,j,2] = G[i,j]
            ans[i,j,3] = 1.0
    return ans
            
            
def ScaleMapping(I, scale = 1.0): # mapping a grayscale image to range 0 ~ 255
    maxv = 0
    minv = 10000
    (height, width, channels) = I.shape
    ans = c.copy(I)
    for i in range(height):
        for j in range(width):
            if I[i,j,0] > maxv:
                maxv = I[i,j,0]
            elif I[i,j,0] < minv:
                minv = I[i,j,0]
    D = maxv - minv
    for i in range(height):
        for j in range(width):
            ans[i,j,0] = (I[i,j,0] - minv) * scale / D
            ans[i,j,1] = ans[i,j,0]
            ans[i,j,2] = ans[i,j,0] 
    return ans
    
    
def Double_Threshold(I,H_ratio):
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
            ans[i,j,3] = 1.0
    return ans

def TrackEdeg(I, O, ii, jj, L_threshold):
    Xindex = [0, -1, -1, -1, 0, 1, 1, 1]
    Yindex = [1, 1, 0, -1, -1, -1, 0, 1]
    for k in range(8):
        i = ii + Xindex[k]
        j = jj + Yindex[k]
        if (I[i,j,0] > L_threshold) and (O[i,j,0] == 0):
            O[i,j,0] = 1
            TrackEdeg(I, O, i, j, L_threshold)
    return
    
    
    
    def GenerateGaussianPyramid(I):
        
    
    
    
