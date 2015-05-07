# M_Dis.py
# Spring 2015 CS_6501 Computer Vision Final
# by Frank
# contains the code for image distance matrix calculation

import numpy as np
import tools as t
import isomap as iso
import mds as m

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
 
#  
#  
#  
# Images = t.ReadImages("ImagesListSy.txt")
   
# ColorDisM = t.ColorDistanceMatrix(Images)
# GrayDisM = t.GrayDistanceMatrix(Images)
#  
# t.SaveMatrix("distanceSy.txt", GrayDisM)
#  
# Images = t.ReadImages("ImagesList3.txt")
#   
# ColorDisM = t.ColorDistanceMatrix(Images)
# # GrayDisM = t.GrayDistanceMatrix(Images)
#  
# t.SaveMatrix("distance3.txt", ColorDisM)

GrayDisM = t.ReadDisMatrix("distanceSy.txt")
 
oset = iso.ObjectSet()

oset.addObjects(GrayDisM)
 
oset.findNeighborhoods(5)
 
print "Neighbor Points generation complete"
 
eset = m.MDS(oset, 3)
 
 
fig = plt.figure()
ax = Axes3D(fig)
fig.set_size_inches(6,6)
for ii in range(len(eset.objectsGraph)):
    plt.plot([eset.objectsGraph[ii].pos[0]], [eset.objectsGraph[ii].pos[1]], "ko", zs=[eset.objectsGraph[ii].pos[2]])
#     for jj in range(ii+1, len(eset.objectsGraph)):
#         if (jj in eset.objectsGraph[ii].neighbors):
#             plt.plot((eset.objectsGraph[ii].pos[0], eset.objectsGraph[jj].pos[0]), (eset.objectsGraph[ii].pos[1], eset.objectsGraph[jj].pos[1]), "k-", zs=(eset.objectsGraph[ii].pos[2], eset.objectsGraph[jj].pos[2]))
 
plt.show()
 
a = 0

 
#  
#   
# IsoDisM = oset.getObjectDeltas()
#     
# Qn1 = t.CreateQMatrix1(IsoDisM)
#   
# Qn2 = t.CreateQMatrix2(IsoDisM)
#      
#      
# print "Q matrix generation complete"
#      
#      
# w1,v1 = np.linalg.eig(Qn1)
#    
# w2,v2 = np.linalg.eig(Qn2)
#    
# n = len(w1)
#    
# w1 = np.real(w1)  
#  
# v1 = np.real(v1)
#    
# D1 = np.zeros([n, n])
#  
# D2 = np.zeros([n, n])
#    
# for i in range(n):
#     D1[i,i] = np.sqrt(w1[i])
#     D2[i,i] = np.sqrt(w2[i])
#    
# Qrank = np.linalg.matrix_rank(Qn1)
#    
# if Qrank < 3:
#     print "Rank of Q is smaller than 3"
#    
# sortw1 = np.argsort(w1)
# sortw2 = np.argsort(w2)
# rev_sortw1 = sortw1[::-1]
# rev_sortw2 = sortw2[::-1]
#    
# dim = 3
#    
# index1 = rev_sortw1[0: dim]
# index2 = rev_sortw2[0: dim]
#     
# # W = np.dot(D, v.T)
#     
# Dr1 = np.zeros([dim,dim])
# Vr1 = np.empty([dim,n])
# Dr2 = np.zeros([dim,dim])
# Vr2 = np.empty([dim,n])
#  
# for i in range(dim):
#     Dr1[i,i] = w1[index1[i]]
#     Vr1[i] = v1[index1[i]]
#     Dr2[i,i] = w2[index2[i]]
#     Vr2[i] = v2[index2[i]]
#    
# D3Cords1 = np.dot(Dr1,Vr1) 
# D3Cords2 = np.dot(Dr2,Vr2) 
#   
# t.outputCords(D3Cords1, "Asymm.obj")
# t.outputCords(D3Cords2, "symm.obj")
    
# D3Cords = D3Cords.T
#     
#    
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#    
# for i in range(n):
#     ax.scatter(D3Cords[0], D3Cords[1], D3Cords[2], c='r')
#        
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
#    
# plt.show()
  


