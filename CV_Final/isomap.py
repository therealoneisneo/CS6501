import numpy
from operator import itemgetter

class IsomapObject:
	name = None
	pos = None
	neighbors = None
	euclidd = None
	geodesicd = None

	def __init__(self, name, pos=None):
		if (pos is not None):
			self.pos = pos.flatten()
		self.name = name
		self.neighbors = set()
		self.euclidd = {}
		self.geodesicd = {}

	def getPos(self):
		return self.pos


class ObjectSet:
	objectsGraph = None

	def __init__(self):
		self.objectsGraph = {}

	def addObject(self, name, pos):
		if (name not in self.objectsGraph):
			self.objectsGraph[name] = IsomapObject(name, pos)
			return True
		else:
			return False

	def addObjects(self, distancemap):
		for ii in range(distancemap.shape[0]):
			self.objectsGraph[ii] = IsomapObject(ii)
			for jj in range(distancemap.shape[1]):
				self.objectsGraph[ii].euclidd[jj] = distancemap[ii, jj]

	def getObjectNames(self):
		return self.objectsGraph.keys()

	def getObject(self, name):
		return self.objectsGraph[name]

	def getObjectDeltas(self):
		out = numpy.empty((len(self.objectsGraph), len(self.objectsGraph)))
		keys = self.objectsGraph.keys()
		for ii in range(len(self.objectsGraph)):
			for jj in range(ii+1):
				out[ii, jj] = self.delta(keys[ii], keys[jj])
			out[:ii, ii] = out[ii, :ii]
		return out

	def findNeighborhoods(self, tgtNeighbors, clear=True):
		edges = []
		keys = self.objectsGraph.keys()

		# generate a list of edges
		# and clear old data if necessary
		for ii in range(len(self.objectsGraph)):
			if clear:
				self.objectsGraph[keys[ii]].neighbors.clear()
				self.objectsGraph[keys[ii]].geodesicd.clear()
			for jj in range(ii):
				edgeLen = self.euclidd(keys[ii], keys[jj])
				edges.append((keys[ii], keys[jj], edgeLen))
		edges.sort(key=itemgetter(2))

		# add the k nearest neighbors for each node. Some nodes may end up with more than k,
		# but all nodes will have at least k neighbors
		ii = 0
		while ii < len(edges):
			thisEdge = edges[ii]
			if (len(self.objectsGraph[thisEdge[0]].neighbors) < tgtNeighbors or len(self.objectsGraph[thisEdge[1]].neighbors) < tgtNeighbors):
				self.objectsGraph[thisEdge[0]].neighbors.add(thisEdge[1])
				self.objectsGraph[thisEdge[1]].neighbors.add(thisEdge[0])
				ii += 1
			else:
				del edges[ii]

		edgelens = [x[2] for x in edges]
		meanEdgelen = numpy.mean(edgelens)
		stddevEdgelen = numpy.std(edgelens)

		for thisEdge in edges:
			if (thisEdge[2] > meanEdgelen + 2* stddevEdgelen):
				self.objectsGraph[thisEdge[0]].neighbors.discard(thisEdge[1])
				self.objectsGraph[thisEdge[1]].neighbors.discard(thisEdge[0])

		for key in self.objectsGraph:
			if len(self.objectsGraph[key].neighbors) == 0:
				del self.objectsGraph[key]


	def euclidd(self, ob1, ob2):
		if (ob2 in self.objectsGraph[ob1].euclidd):
			return self.objectsGraph[ob1].euclidd[ob2]
		else:
			dist = numpy.sqrt( numpy.sum( numpy.power((self.objectsGraph[ob1].pos - self.objectsGraph[ob2].pos),2)) )
			self.objectsGraph[ob1].euclidd[ob2] = dist
			self.objectsGraph[ob2].euclidd[ob1] = dist
			return dist

	def buildGeodesicCache(self):
		keys = self.objectsGraph.keys()
		for ii in range(len(self.objectsGraph)):
			keyii = keys[ii]
			self.objectsGraph[keyii].geodesicd.clear()
			self.objectsGraph[keyii].geodesicd[keyii] = 0
			for jj in range(ii):
				if (keys[jj] in self.objectsGraph[keyii].neighbors):
					self.objectsGraph[keyii].geodesicd[keys[jj]] = self.euclidd(keyii, keys[jj])
					self.objectsGraph[keys[jj]].geodesicd[keyii] = self.euclidd(keyii, keys[jj])
				else:
					self.objectsGraph[keyii].geodesicd[keys[jj]] = float('inf')
					self.objectsGraph[keys[jj]].geodesicd[keyii] = float('inf')
		for kk in range(len(self.objectsGraph)):
			keykk = keys[kk]
			for ii in range(len(self.objectsGraph)):
				keyii = keys[ii]
				for jj in range(len(self.objectsGraph)):
					keyjj = keys[jj]
					distkk = self.objectsGraph[keyii].geodesicd[keykk] + self.objectsGraph[keykk].geodesicd[keyjj]
					if (self.objectsGraph[keyii].geodesicd[keyjj] > distkk):
						self.objectsGraph[keyii].geodesicd[keyjj] = distkk
						self.objectsGraph[keyjj].geodesicd[keyii] = distkk

	def geodesicd(self, ob1, ob2):
		if (ob2 in self.objectsGraph[ob1].geodesicd):
			return self.objectsGraph[ob1].geodesicd[ob2]
		else:
			return float('inf')

	def delta(self, ob1, ob2):
		return (self.euclidd(ob1, ob2) + self.geodesicd(ob1, ob2))/2.0

if __name__ == "__main__":
	dims = 2
	dimrange = (0,100)
	pointcount = 400
	avgNeighbors = 10

	import numpy.random
	import matplotlib.pyplot as plt
	import matplotlib.lines
	fig, ax = plt.subplots()
	fig.set_size_inches(6,6)

	oset = ObjectSet()
	for ii in range(pointcount):
		oset.addObject(ii, numpy.random.rand(dims) * (dimrange[1] - dimrange[0]) + dimrange[0])
	oset.findNeighborhoods(avgNeighbors)
	oset.buildGeodesicCache()
	for ii in range(len(oset.objectsGraph)):
		plt.plot(oset.objectsGraph[ii].pos[0], oset.objectsGraph[ii].pos[1], "ko")
		for jj in range(ii+1, len(oset.objectsGraph)):
			if (jj in oset.objectsGraph[ii].neighbors):
				plt.plot((oset.objectsGraph[ii].pos[0], oset.objectsGraph[jj].pos[0]), (oset.objectsGraph[ii].pos[1], oset.objectsGraph[jj].pos[1]), "k-")

	plt.show()






