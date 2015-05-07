import numpy as np
import copy as c
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay


def RotateCords(Cords, angle, axis):
	angle = float(angle) * np.pi / 180
	if axis == 'x':
		RotM = [[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]]
	elif axis == 'y':
		RotM = [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
	elif axis == 'z':
		RotM = [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
	ans = np.dot(RotM, Cords)
	return ans

def SphereToGlobal(Cords):
	radius = Cords[0]
	theta = Cords[1]
	phi = Cords[2]
	x = radius * np.sin(theta) * np.cos(phi)
	y = radius * np.sin(theta) * np.sin(phi)
	z = radius * np.cos(theta)
	ans = np.array([x,y,z])
	return ans

def GlobalToSphere(Cords):
	x = Cords[0]
	y = Cords[1]
	z = Cords[2]
	
	radius = np.linalg.norm(np.array([x, y, z]))
	theta = 0
	phi = 0
	
	if z == 0:
		theta = np.pi / 2
	else:
		theta = np.arccos(float(z) / radius)
	if x == 0:
		if y > 0:
			phi = np.pi / 2
		if y < 0:
			phi = np.pi * 3 / 2
	if x > 0 and y >= 0:
		phi = np.arctan(float(y)/x)
	if x < 0 and y >= 0:
		phi = np.pi + np.arctan(float(y) / x)
	if x < 0 and y < 0:
		phi = np.arctan(float(y) / x) + np.pi
	if x > 0 and y < 0:
		phi = np.pi * 2 + np.arctan(float(y)/x)
	
	ans = np.array([radius, theta, phi])
	return ans


def SphereDataGen(num, radius = 1 , angle = 90, arms = 4): 
	# generate a "num" of spherical data with "radius", distibuted in "angle" and "arms"
	angle = float(angle) * np.pi / 180
	ArmPoints = int(num / arms) + 1
	Cords = []
	ThetaDelta = angle / ArmPoints 
	PhiDelta = 2 * np.pi / arms
	
	temp = []
	temp.append(0)
	temp.append(0)
	temp.append(radius)
	Cords.append(temp)
	
	for i in range(arms):
		phi = i * PhiDelta
		for j in range(1, ArmPoints):
			theta = j * ThetaDelta
			temp = []
			(x, y, z) = SphereToGlobal([radius, theta, phi])
			temp.append(x)
			temp.append(y)
			temp.append(z)
			Cords.append(temp)
	Cords = np.array(Cords).T
	Cords = RotateCords(Cords, 35, 'x')
	Cords = RotateCords(Cords, 28, 'y')
	Cords = RotateCords(Cords, 60, 'z')
	return Cords.T



# def Gridlize(SphereCords, pieces): # divide the sphere (centered a 0,0,0) in to "pieces" of grids
#     # using 4 points to indicate a piece, if there are 2 points are the same, then it is a triangle
#     # "pieces" has to be even number
#     pieces /= 2

def CenterOfMass(Cords):#center of mass
	MCenter = np.array([0, 0, 0])
	for i in range(Cords.shape[0]):
		MCenter = np.add(MCenter, Cords[i])
	MCenter = MCenter.astype(float)
	ans = np.linalg.norm(Cords[0]) * MCenter / np.linalg.norm(MCenter)
	return ans 
	

def Rectify (Cords, dirVec): # rectify Sphere cords with "dirVec" oriantation to z axis
	ans = c.copy(Cords)
	SphereDir = GlobalToSphere(dirVec)
	phi = - SphereDir[2] * 180 / np.pi
	theta = - SphereDir[1] * 180 / np.pi
	ans = RotateCords(ans, phi, 'z')
	ans = RotateCords(ans, theta, 'y')
	return ans

def UnfoldSphere(Cords, figRadius = 0):# unfold a Sphere into a 2D circle
	SCords = c.copy(Cords)
	ans = np.empty([SCords.shape[0], 2])
	for i in range(SCords.shape[0]):
		SCords[i] = GlobalToSphere(SCords[i])
		ans[i, 0] = np.cos(SCords[i, 2]) * SCords[i, 1]
		ans[i, 1] = np.sin(SCords[i, 2]) * SCords[i, 1]
#         if (SCords[i, 2] <= (2 * np.pi) and SCords[i, 2] > (3.0 / 2  * np.pi)) or (SCords[i, 2] <= (1.0 / 2 * np.pi) and SCords[i, 2] > 0):
#             ans[i, 0] = np.cos(SCords[i, 2]) * SCords[i, 1]
#             ans[i, 1] = np.sin(SCords[i, 2]) * SCords[i, 1]
#     a = SCords.T
	return ans

def Interpolate(paneldata, x, y):
	cord = np.array([x,y])
	n = len(paneldata)
	tri = Delaunay(paneldata)
	triangleindex = tri.find_simplex(cord)
	intercords = tri.simplices[triangleindex]
	index1 = np.empty(2)
	index2 = np.empty(2)
	index3 = np.empty(2)
	if triangleindex >= 0:
		index1[0] = intercords[0]
		index2[0] = intercords[1]
		index3[0] = intercords[2]
		x1 = tri.points[intercords[0]][0]
		y1 = tri.points[intercords[0]][1]
		x2 = tri.points[intercords[1]][0]
		y2 = tri.points[intercords[1]][1]
		x3 = tri.points[intercords[2]][0]
		y3 = tri.points[intercords[2]][1]
		
		index1[1] = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3))
		index2[1] = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3))
		index3[1] = 1 - index1[1] - index2[1]
	else:
		index1[0] = -1
		index1[1] = 10000000
		index2[0] = -1
		index2[1] = 10000000
		index3[0] = 1
		index3[1] = 0
		distance = np.empty(n)
		for i in range(n):
			distance[i] = np.linalg.norm(np.subtract(cord, paneldata[i]))
			if distance[i] < index1[1]:
				index2[0] = index1[0]
				index2[1] = index1[1]
				index1[0] = i
				index1[1] = distance[i]
			elif distance[i] < index2[1]:
				index2[0] = i
				index2[1] = distance[i]
		base = index1[1] + index2[1]
		index1[1] /= base
		index2[1] /= base
	ans = np.array([index1, index2, index3])        
	return ans





if __name__ == "__main__":

	Cords = SphereDataGen( 600, 20, 130, 20)

	CMass = CenterOfMass(Cords)
	 
	RctCords = Rectify(Cords.T, CMass).T
	 
	Panel = UnfoldSphere(RctCords)







	test1 = Interpolate(Panel, 0.5, 0.6)
	test2 = Interpolate(Panel, -0.3, -0.6)
	 
	x1 = Panel[int(test1[0, 0])] * test1[0, 1] + Panel[int(test1[1, 0])] * test1[1, 1] + Panel[int(test1[2, 0])] * test1[2, 1]
	x2 = Panel[int(test2[0, 0])] * test2[0, 1] + Panel[int(test2[1, 0])] * test2[1, 1] + Panel[int(test2[2, 0])] * test2[2, 1]

	 
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
			
	for i in range(Cords.shape[0]):
		ax.scatter(Cords[i, 0], Cords[i, 1], Cords[i, 2], c='r')
	ax.scatter(CMass[0], CMass[1], CMass[2], c='y')
				
	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')





	# fig1 = plt.figure()
	# ax1 = fig1.add_subplot(111, projection='3d')
	#       
	# for i in range(RctCords.shape[0]):
	#     ax1.scatter(RctCords[i, 0], RctCords[i, 1], RctCords[i, 2], c='r')
	# # ax.scatter(CMass[0], CMass[1], CMass[2], c='y')
	#           
	# ax1.set_xlabel('X Label')
	# ax1.set_ylabel('Y Label')
	# ax1.set_zlabel('Z Label')



	fig2 = plt.figure()
	ax2 = fig2.add_subplot(111, projection='3d')
			
	for i in range(Panel.shape[0]):
		ax2.scatter(Panel[i, 0], Panel[i, 1], 0, c='r')
		 
	ax2.scatter(0.5, 0.6, 0.5, c='g')
	ax2.scatter(-0.3, -0.6, 0.5, c='g')
	 
	ax2.scatter(x1[0], x1[1], 1, c = 'y')
	ax2.scatter(x2[0], x2[1], 1, c = 'y')
	# ax.scatter(CMass[0], CMass[1], CMass[2], c='y')
				
	ax2.set_xlabel('X Label')
	ax2.set_ylabel('Y Label')
	ax2.set_ylabel('Z Label')



	# t.outputCords(Cords.T, "Sphere.obj")
	# 
	# fig2 = plt.figure()
	# ax2 = fig2.add_subplot(111, projection='3d')
	#       
	# for i in range(test.shape[0]):
	#     ax2.scatter(test[i, 0], test[i, 1], test[i, 2], c='r')
	# # ax.scatter(CMass[0], CMass[1], CMass[2], c='y')
	#           
	# ax1.set_xlabel('X Label')
	# ax1.set_ylabel('Y Label')
	# ax1.set_zlabel('Z Label')


	plt.show()

	a= 0
				
		