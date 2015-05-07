# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 10:26:17 2015

@author: LH
"""
from svd_FitHypersphere import fit_hypersphere
methods = ["Hyper", "Taubin", "Pratt"]                  # for svd_FitHypersphere
from math import sin, cos, pi, exp, log, sqrt, pow
import numpy as np
import SphereTop as st
import isomap as iso



from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def sphFitting(sphere):
    
    datax,datay,dataz = zip(*sphere)    
    fit = fit_hypersphere(sphere,methods[0])
    
    sphr = fit[0]
    sphx = fit[1][0]
    sphy = fit[1][1]
    sphz = fit[1][2]
    
    vectorx = [x - sphx for x in datax]
    vectory = [y - sphy for y in datay]
    vectorz= [z - sphz for z in dataz]

    vectorlist = zip(*[vectorx, vectory, vectorz])
    coordonsph = [[(vector[0] * sphr /sqrt(pow(vector[0],2)+pow(vector[1],2)+pow(vector[2],2)) + sphx),(vector[1] * sphr /sqrt(pow(vector[0],2)+pow(vector[1],2)+pow(vector[2],2)) + sphy),(vector[2] * sphr /sqrt(pow(vector[0],2)+pow(vector[1],2)+pow(vector[2],2))+ sphz) ]for vector in vectorlist]
        
    xonsph, yonsph, zonsph = zip(*coordonsph)
    retVal = np.asarray(coordonsph)
    return retVal
    

def splitLine(line):
    elements = line.split(',')
    retVal = [float(elements[1]),float(elements[2]),float(elements[3].strip())]
    return retVal



def unfold(sphere, geoM):
# f = open("sphcordsprims200.csv")
# sphere = [splitLine(line) for line in f]
    fitting = sphFitting(sphere)
#     cmass = st.CenterOfMass(fitting)
    cmass = st.Topcords(geoM, fitting)
    Re_fitting = st.Rectify(fitting, cmass)
    panel = st.UnfoldSphere(Re_fitting)
    return panel


def EDistanceOfVecs(VecM):
    (num, dim) = VecM.shape
    ans = np.empty([num,num])
    for i in range(num):
        for j in range(i, num):
            temp = np.subtract(VecM[i], VecM[j])
            ans[i, j] = np.linalg.norm(temp)
            ans[j, i] = ans[i, j]
    return ans



if __name__ == "__main__":
    f = open("sphcordsprims200.csv")
    sphere = [splitLine(line) for line in f]
    
    fitting = sphFitting(sphere)
#     cmass = st.CenterOfMass(fitting)
    
    print 1
    
    DisM = EDistanceOfVecs(fitting)
    
    print 2
    oset = iso.ObjectSet()

    print 3
    oset.addObjects(DisM)
    
    print 4
 
    oset.findNeighborhoods(25)
    
    print 5
    oset.buildGeodesicCache()
    
    
    print 6
    GeoDisM = oset.getObjectDeltas()
    
    print 7
    CMass1 = st.CenterOfMass(fitting)
    
    print 8
#     cmass = st.Topcords(GeoDisM, fitting)
    
    length = 10000000000
    index = -1
    center = np.array([0, 3200, -9500])
    for i in range(len(fitting)):
        dis = np.linalg.norm(np.subtract(center, fitting[i]))
        if dis < length:
            length = dis
            index = i
    cmass = fitting[index]
#     RctCords = st.Rectify(fitting, CMass)
#      
#     Panel = st.UnfoldSphere(RctCords)

    
    
#     cmass = st.Topcords(geoM, fitting)
    Re_fitting = st.Rectify(fitting, cmass)
    panel = st.UnfoldSphere(Re_fitting)





# f.close()



    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
             
    for i in range(panel.shape[0]):
        ax.scatter(panel[i, 0], panel[i, 1], 0, c='r')
    # ax.scatter(cMass[0], CMass[1], CMass[2], c='y')
                 
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
     
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
             
    for i in range(len(fitting)):
        ax1.scatter(fitting[i][0], fitting[i][1], fitting[i][2], c='r')
    # ax.scatter(cMass[0], CMass[1], CMass[2], c='y')
                 
    ax1.set_xlabel('X Label')
    ax1.set_ylabel('Y Label')
    ax1.set_zlabel('Z Label')
     
     
     
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
             
    for i in range(Re_fitting.shape[0]):
        ax2.scatter(Re_fitting[i, 0], Re_fitting[i, 1], Re_fitting[i, 2], c='r')
    ax2.scatter(cmass[0], cmass[1], cmass[2], c='y')
                 
    ax2.set_xlabel('X Label')
    ax2.set_ylabel('Y Label')
    ax2.set_zlabel('Z Label')
     
    plt.show()