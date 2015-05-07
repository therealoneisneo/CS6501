import numpy, numpy.linalg

class EmbeddedObject:
	name = None
	pos = None
	deltas = None

	def __init__(self, name, pos):
		self.name = name
		self.pos = pos

	def getPos(self):
		return self.pos

class ObjectSet:
	objectsGraph = None

	def __init__(self):
		self.objectsGraph = {}

	def addObject(self, name, pos):
		if (name not in self.objectsGraph):
			self.objectsGraph[name] = EmbeddedObject(name, pos)
			return True
		else:
			return False

	def getObjectNames(self):
		return self.objectsGraph.keys()

	def getObject(self, name):
		return self.objectsGraph[name]

	def getObjectPositions(self):
		dims = self.objectsGraph[self.objectsGraph.keys()[0]].pos.shape[0]
		out = numpy.empty((len(self.objectsGraph),dims))
		for ii in range(len(self.objectsGraph)):
			out[ii,:] = self.objectsGraph[ii].pos
		return out

def MDS(objset, tgtdims):
	objNames = objset.getObjectNames()
	N = len(objNames)
	if (tgtdims > N):
		return False

	Q = objset.getObjectDeltas() ** 2

	Q = -0.5 * (Q - (1.0/N) * (numpy.sum(Q, axis=0, keepdims=True) + numpy.sum(Q, axis=1, keepdims=True)) + numpy.sum(Q, keepdims=True)/float(N**2))
	vals, vecs = numpy.linalg.eig(Q)

	idx = vals.argsort()[::-1]
	vals = vals[idx]
	vecs = vecs[:,idx]

	coords = vecs[:,:tgtdims] * numpy.sqrt(numpy.sum(vals[:tgtdims]))

	outputSet = ObjectSet()
	for ii in range(N):
		outputSet.addObject(objNames[ii], coords[ii, :])

	return outputSet

if __name__ == "__main__":
	import isomap

	dims = 3
	tgtdims = 3
	dimrange = (0,100)
	pointcount = 250
	avgNeighbors = 10

	import numpy.random
	import matplotlib.pyplot as plt
	import matplotlib.lines
	from mpl_toolkits.mplot3d import axes3d, Axes3D
	fig = plt.figure()
	ax = Axes3D(fig)
	fig.set_size_inches(6,6)


	print "Finding Neighborhoods"
	iset = isomap.ObjectSet()
	for ii in range(pointcount):
		phi = numpy.random.rand() * numpy.pi * 2.0
		theta = numpy.arccos(2.0 * numpy.random.rand() - 1)
		r = numpy.random.rand() * 10 + 95
		xyz = numpy.empty((3,))
		xyz[0] = r*numpy.cos(theta)*numpy.cos(phi)
		xyz[1] = r*numpy.sin(theta)*numpy.cos(phi)
		xyz[2] = r * numpy.sin(phi)
		iset.addObject(ii, xyz)
		#iset.addObject(ii, numpy.random.rand(dims) * (dimrange[1] - dimrange[0]) + dimrange[0])
	iset.findNeighborhoods(avgNeighbors)
	iset.buildGeodesicCache()
	
	for ii in range(len(iset.objectsGraph)):
		plt.plot([iset.objectsGraph[ii].pos[0]], [iset.objectsGraph[ii].pos[1]], "ko", zs=[iset.objectsGraph[ii].pos[2]])
		for jj in range(ii+1, len(iset.objectsGraph)):
			if (jj in iset.objectsGraph[ii].neighbors):
				plt.plot((iset.objectsGraph[ii].pos[0], iset.objectsGraph[jj].pos[0]), (iset.objectsGraph[ii].pos[1], iset.objectsGraph[jj].pos[1]), "k-", zs=(iset.objectsGraph[ii].pos[2], iset.objectsGraph[jj].pos[2]))
	
	plt.show()

	print "Reducing Dimensionality"
	oset = MDS(iset, tgtdims)

	fig = plt.figure()
	ax = Axes3D(fig)
	fig.set_size_inches(6,6)
	for ii in range(len(oset.objectsGraph)):
		plt.plot([oset.objectsGraph[ii].pos[0]], [oset.objectsGraph[ii].pos[1]], "ko", zs=[oset.objectsGraph[ii].pos[2]])
		for jj in range(ii+1, len(oset.objectsGraph)):
			if (jj in iset.objectsGraph[ii].neighbors):
				plt.plot((oset.objectsGraph[ii].pos[0], oset.objectsGraph[jj].pos[0]), (oset.objectsGraph[ii].pos[1], oset.objectsGraph[jj].pos[1]), "k-", zs=(oset.objectsGraph[ii].pos[2], oset.objectsGraph[jj].pos[2]))

	plt.show()