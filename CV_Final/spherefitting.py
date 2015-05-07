# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 10:26:17 2015

@author: LH
"""
from svd_FitHypersphere import fit_hypersphere
methods = ["Hyper", "Taubin", "Pratt"]                  # for svd_FitHypersphere
from math import sin, cos, pi, exp, log, sqrt, pow
import numpy as np
import spheremanipulation as st

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






if __name__ == "__main__":
    f = open("sphcordsprims200.csv")
    sphere = [splitLine(line) for line in f]
    
    fitting = sphFitting(sphere)
    #     cmass = st.CenterOfMass(fitting)
    cmass = st.Topcords(geoM, fitting)
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
             
    for i in range(len(sphere)):
        ax1.scatter(sphere[i][0], sphere[i][1], sphere[i][2], c='r')
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