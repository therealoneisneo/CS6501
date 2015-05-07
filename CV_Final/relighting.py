import sys, getopt, math
import numpy, scipy.misc
import isomap
import mds
import spheremanipulation
import spherefitting

if __name__ == "__main__":
	filetype = "png"
	sourcedir = "images"
	image_width = 1024
	num_neighbors = 10
	output_file = "embedding.csv"
	show_result = False

	opts, args = getopt.getopt(sys.argv[1:],"s:t:w:n:o:d",[])

	for opt, arg in opts:
		if opt == '-s':
			sourcedir = arg
		elif opt == '-t':
			filetype = arg
		elif opt == '-w':
			image_width = int(arg)
		elif opt == '-n':
			num_neighbors = int(arg)
		elif opt == "-o":
			output_file = arg
		elif opt == "-d":
			show_result = True

	from os import listdir
	from os.path import isfile, join
	onlyfiles = [ f for f in listdir(sourcedir) if (isfile(join(sourcedir,f)) and f[0] != '.' and f[-4:] == "." + filetype) ]

	images = []
	print "loading images: " + (" " * int(math.ceil(math.log(len(onlyfiles), 10)))),

	for inputfile in onlyfiles:
		print "\b" * (int(math.ceil(math.log(len(images) + 1, 10))) + 1) + str(len(images) + 1),
		sys.stdout.flush()
		im = scipy.misc.imread(join(sourcedir,inputfile), flatten=True)/255.0
		images.append((inputfile, scipy.misc.imresize( im, (image_width, int( im.shape[0]*(float(image_width)/im.shape[1]) )), interp='bilinear' )))

	print "\ncalculating raw distances: " + (" " * int(math.ceil(math.log(len(images), 10)))),
	raw_distances = numpy.empty((len(images), len(images)))
	for ii in range(len(images)):
		print "\b" * (int(math.ceil(math.log(ii + 1, 10))) + 1) + str(ii + 1),
		sys.stdout.flush()
		raw_distances[ii,ii] = 0.0
		for jj in range(ii):
			dist = numpy.sum(numpy.power(images[ii][1] - images[jj][1], 2))
			raw_distances[ii,jj] = dist
			raw_distances[jj,ii] = dist

	print "\nfinding neighborhoods"
	isomap_set = isomap.ObjectSet()
	isomap_set.addObjects(raw_distances)
	isomap_set.findNeighborhoods(num_neighbors)
	print "calculating geodesic distances"
	isomap_set.buildGeodesicCache()
	print "reducing dimensionality"
	embedded_set = mds.MDS(isomap_set, 3)
	three_coords = embedded_set.getObjectPositions()
	print "unfolding sphere"
	two_coords = spherefitting.unfold(three_coords, isomap_set.getObjectDeltas())

	with open(output_file, 'w') as output:
		for ii in range(len(embedded_set.objectsGraph)):
			output.write(images[ii][0] + "," + str(two_coords[ii, 0]) + "," + str(two_coords[ii, 1]) + "\n")

	#if show_result:
	#	fig = plt.figure()
	#	ax = Axes3D(fig)
	#	fig.set_size_inches(6,6)
	#	for ii in range(len(embedded_set.objectsGraph)):
	#		plt.plot([embedded_set.objectsGraph[ii].pos[0]], [embedded_set.objectsGraph[ii].pos[1]], "ko", zs=[embedded_set.objectsGraph[ii].pos[2]])
	#		for jj in range(ii+1, len(embedded_set.objectsGraph)):
	#			if (jj in iset.objectsGraph[ii].neighbors):
	#				plt.plot((embedded_set.objectsGraph[ii].pos[0], embedded_set.objectsGraph[jj].pos[0]), (embedded_set.objectsGraph[ii].pos[1], embedded_set.objectsGraph[jj].pos[1]), "k-", zs=(embedded_set.objectsGraph[ii].pos[2], embedded_set.objectsGraph[jj].pos[2]))
	#
	#	plt.show()
