#try graph matching 

from PIL import Image
import numpy as np # numpy backend
import pygmtools as pygm
import matplotlib.pyplot as plt # for plotting
from matplotlib.patches import ConnectionPatch # for plotting matching result
import networkx as nx # for plotting graphs
pygm.set_backend('numpy') # set default backend for pygmtools

import functools

import numpy as np
import skimage.io
import tifffile
import matplotlib.pyplot as plt
import scipy.spatial.distance
import matplotlib.pyplot as plt 
import itertools
import scipy.stats as st
import scipy.signal 

import skimage.future
import skimage.graph
import scipy.ndimage
import skimage.measure

#https://forum.image.sc/t/reading-pixel-size-from-image-file-with-python/74798/5
def read_tiff_voxel_size(file_path):
    """
    Implemented based on information found in https://pypi.org/project/tifffile
    """

    def _xy_voxel_size(tags, key):
        assert key in ['XResolution', 'YResolution']
        if key in tags:
            num_pixels, units = tags[key].value
            return units / num_pixels
        # return default
        return 1.

    with tifffile.TiffFile(file_path) as tiff:
        image_metadata = tiff.imagej_metadata
        if image_metadata is not None:
            z = image_metadata.get('spacing', 1.)
        else:
            # default voxel size
            z = 1.

        tags = tiff.pages[0].tags
        # parse X, Y resolution
        y = _xy_voxel_size(tags, 'YResolution')
        x = _xy_voxel_size(tags, 'XResolution')
        # return voxel size
        return [z, y, x]

def label2Contours2D(ImgL1):

	region_borders = scipy.ndimage.morphological_gradient(ImgL1,
                                            footprint=[[0, 0, 0],
                                                       [0, 1, 0],
                                                       [0, 0, 0]])
	yxB = np.argwhere(region_borders!=0)
	region_borders[yxB[:,0],yxB[:,1]]=1
	return region_borders 

def my_node_aff_fn(featAll1, featAll2): # feat1 has shape (n_1, f), feat2 has shape (n_2, f)
	#embed()                     # use functools.partial if you want to specify sigma value
		
	feat1 = np.expand_dims(featAll1[...,0], axis=2)
	feat2 = np.expand_dims(featAll2[...,0], axis=1)
	m1 = ((feat1 - feat2) ** 2) / (feat1 + feat2)
	
	feat1 = np.expand_dims(featAll1[...,1], axis=2)/feat1.sum()
	feat2 = np.expand_dims(featAll2[...,1], axis=1)/feat2.sum()
	m2 = ((feat1 - feat2) ** 2) #/ #(feat1.sum() + )
	return np.exp(-(m1+(m2-m2.min())/m2.max()))
	
def my_edge_aff_fn(featAll1, featAll2): # feat1 has shape (n_1, f), feat2 has shape (n_2, f)
	#embed()                     # use functools.partial if you want to specify sigma value
		
	feat1 = np.expand_dims(featAll1[...,0], axis=2)
	feat2 = np.expand_dims(featAll2[...,0], axis=1)
	m1 = ((feat1/feat1.sum() - feat2/feat2.sum()) ** 2) # / (feat1 + feat2)
	
	feat1 = np.expand_dims(featAll1[...,1], axis=2)
	feat2 = np.expand_dims(featAll2[...,1], axis=1)
	m2 = ((feat1 - feat2) ** 2) / feat1.max()
	return np.exp(-(((m1-m1.min())/m1.max())+((m2-m2.min())/m2.max())))
	
def extractOrganOrientation(img, bgLabel):
	borderLabels = np.unique(np.concatenate([np.unique(img[:,-1]),np.unique(img[-1,:]),np.unique(img[0,:]),np.unique(img[:,0])]))
	imBin = (~np.isin(img,borderLabels )).astype(np.uint8)
	xyz = np.argwhere(imBin)
	Ci = np.mean(xyz, axis=0)
	yxz_centered = xyz - Ci

	Si = np.dot(yxz_centered.T, yxz_centered) / yxz_centered.shape[0] #covariance

	lr, vr = np.linalg.eigh(Si)
	#eigenV.append(vr)
		
	return vr[:,np.argmax(lr)]



segmentationFileName1 = "xxx.npy"#path to a npy (segmentation)

segmentationFileName2 = "yyy.npy" ## path to a second tif (segmentation)
outImgFolder = segmentationFileName1[:segmentationFileName1.rfind(".")]
dz,dx,dy =  [1,1,1] #read_tiff_voxel_size(segmentationFileName1)

label1 = [95] #label in segmentation 1
label2 = [153] # label in segmentation 2
seeds = [[95,153]]


imgSeg1 = np.load(segmentationFileName1,allow_pickle=True) #skimage.io.imread(segmentationFileName1)
imgSeg2 = np.load(segmentationFileName2,allow_pickle=True) #skimage.io.imread(segmentationFileName2)

Image.fromarray(imgSeg1).save(segmentationFileName1[:segmentationFileName1.rfind(".")]+".tif")
Image.fromarray(imgSeg2).save(segmentationFileName2[:segmentationFileName2.rfind(".")]+".tif")

imC1 = label2Contours2D(imgSeg1)
imC2 = label2Contours2D(imgSeg2)

vecOrgan1 = [0,1]#extractOrganOrientation(imgSeg1, np.median(imgSeg1))
vecOrgan2 = [0,1]#extractOrganOrientation(imgSeg2, np.median(imgSeg2))

rag1 = skimage.graph.rag_boundary(imgSeg1.astype(np.int64), imC1.astype(float))
rag2 = skimage.graph.rag_boundary(imgSeg2.astype(np.int64), imC2.astype(float))

rp1 = skimage.measure.regionprops(imgSeg1.astype(np.uint16))
rp2 = skimage.measure.regionprops(imgSeg2.astype(np.uint16))
#rp1[label1].label #rp1[label1].axis_major_length
labels1 = np.array([e.label for e in rp1])
labels2 = np.array([e.label for e in rp2])

iL1 = np.zeros(labels1.max()+1,dtype= int)
iL1[labels1] = np.arange(len(labels1))
iL2 = np.zeros(labels2.max()+1,dtype= int)
iL2[labels2] = np.arange(len(labels2))

showOn = 0
g1 = nx.Graph(rag1)
g1.remove_node(0)
g2 = nx.Graph(rag2)
g2.remove_node(0)


### iterative solving second trial : with iteration on the graph instead of the affinity matrix 
assignedNodes1 = [seeds[0][0]]
assignedNodes2 = [seeds[0][1]]
assignedNumberPrev =  -1
iteration = 0
while (len(assignedNodes1)!= len(labels1)) and (len(assignedNodes2)!= len(labels2)) and (len(assignedNodes1) != assignedNumberPrev):
	assignedNumberPrev = len(assignedNodes1)
	num_seeds = len(assignedNodes1)
	## add first round of neighbooring nodes
	toAssigned1 = []
	toAssigned2 = []
	for eachSeed in range(len(assignedNodes1)):
		toAssigned1 = toAssigned1+list(np.array(list(g1.edges(assignedNodes1[eachSeed])))[:,1])
		toAssigned2 = toAssigned2+list(np.array(list(g2.edges(assignedNodes2[eachSeed])))[:,1])
	toAssigned1 = np.unique(toAssigned1+assignedNodes1)
	toAssigned2 = np.unique(toAssigned2+assignedNodes2)
	
	## select the subgraphs
	g1Sub = g1.subgraph(toAssigned1) ##there can remove the already assigned nodes except the ones  of the last iteration TODO
	g2Sub = g2.subgraph(toAssigned2)
	
	labels1Sub = list(g1Sub.nodes())
	labels2Sub = list(g2Sub.nodes())
	siL1 = np.zeros(max(labels1Sub)+1,dtype= int)
	siL1[labels1Sub] = np.arange(len(labels1Sub))
	siL2 = np.zeros(max(labels2Sub)+1,dtype= int)
	siL2[labels2Sub] = np.arange(len(labels2Sub))

	seed_subMat = np.zeros((len(labels1Sub), len(labels2Sub)))
	posSeedL1 = siL1[np.array(assignedNodes1)]
	posSeedL2 = siL2[np.array(assignedNodes2)]
	seed_subMat[posSeedL1,posSeedL2] = 1

	pos1 = nx.spring_layout(g1Sub)
	pos2 = nx.spring_layout(g2Sub)
	if showOn:
		plt.figure(figsize=(8, 4))
		ax1 = plt.subplot(1, 2, 1)
		plt.title('Graph 1')
		nx.draw_networkx(g1Sub, pos=pos1)
		ax2 = plt.subplot(1, 2, 2)
		plt.title('Graph 2')
		nx.draw_networkx(g2Sub, pos=pos2)
		for i in range(num_seeds):
			'''
			j = np.argmax(seed_mat[posSeedL1[i]]).item()
			j = labels2Sub[j]
			i = assignedNodes1[i] #int(posSeedL1) label
			'''
			
			con = ConnectionPatch(xyA=pos1[assignedNodes1[i]], xyB=pos2[assignedNodes2[i]], coordsA="data", coordsB="data",
					  axesA=ax1, axesB=ax2, color="blue")
			plt.gca().add_artist(con)
		plt.show()


	conn1, edge1 = pygm.utils.dense_to_sparse(nx.to_numpy_array(g1Sub))  ## conn is indexes and not labels
	conn2, edge2 = pygm.utils.dense_to_sparse(nx.to_numpy_array(g2Sub))

	if len(conn1) == 0:
		affinityMatrix1 = np.zeros((len(labels1Sub),len(labels1Sub)))
		edges1 = list(g1Sub.edges())
		for iLab1 in range(len(edges1)):
			i1 = np.squeeze(np.argwhere(labels1Sub==edges1[iLab1][0]))
			i2 = np.squeeze(np.argwhere(labels1Sub==edges1[iLab1][1]))
			affinityMatrix1[i1,i2] = 1
			affinityMatrix1[i2,i1] = 1
				
		conn1, edge1 = pygm.utils.dense_to_sparse(affinityMatrix1) 
	if len(conn2)== 0:
		affinityMatrix2 = np.zeros((len(labels2Sub),len(labels2Sub)))
		edges2 = list(g2Sub.edges())
		for iLab2 in range(len(edges2)):
			i1 = np.squeeze(np.argwhere(labels2Sub==edges2[iLab2][0]))
			i2 = np.squeeze(np.argwhere(labels2Sub==edges2[iLab2][1]))
			affinityMatrix2[i1,i2] = 1
			affinityMatrix2[i2,i1] = 1
				
		conn2, edge2 = pygm.utils.dense_to_sparse(affinityMatrix2) 	
	edgeFeatures1 = []
	vecImg = vecOrgan1 # [0,0,1]
	for link in conn1:
		l1 = labels1Sub[link[0]]#toAssigned1[link[0]] #
		l2 = labels1Sub[link[1]]#toAssigned1[link[1]] #
		iNeOri = int(np.squeeze(np.argwhere(labels1==l1)))
		iNe = int(np.squeeze(np.argwhere(labels1==l2)))
		vec = np.array([rp1[iNe].centroid[0]*dy-rp1[iNeOri].centroid[0]*dy,rp1[iNe].centroid[1]*dx-rp1[iNeOri].centroid[1]*dx])
		normVec =  np.sqrt(vec[0]**2 +vec[1]**2)
		angles = np.arccos(np.clip(np.dot(vec.T,vecImg)/ normVec,-1.0, 1.0))
			
		edgeFeatures1.append([rag1[l1][l2]['count'],np.rad2deg(angles)])

	edgeFeatures1 = np.array(edgeFeatures1)


	vecImg = vecOrgan2 #[0,0,1]
	edgeFeatures2 = []
	for link in conn2:
		l1 = labels2Sub[link[0]]#toAssigned2[link[0]] #
		l2 = labels2Sub[link[1]]#toAssigned2[link[1]] #
		iNeOri = int(np.squeeze(np.argwhere(labels2==l1)))
		iNe = int(np.squeeze(np.argwhere(labels2==l2)))
		vec = np.array([rp2[iNe].centroid[0]*dy-rp2[iNeOri].centroid[0]*dy,rp2[iNe].centroid[1]*dx-rp2[iNeOri].centroid[1]*dx])
		normVec =  np.sqrt(vec[0]**2 +vec[1]**2)
		angles = np.arccos(np.clip(np.dot(vec.T,vecImg)/ normVec,-1.0, 1.0))
			
		edgeFeatures2.append([rag2[l1][l2]['count'],np.rad2deg(angles)])

	edgeFeatures2 = np.array(edgeFeatures2)

	### node features
	node_feat1 = []
	for l in  labels1Sub:
		iNe = iL1[l]# int(np.squeeze(np.argwhere(labels1==l)))
		node_feat1.append([len(g1.edges(l)),rp1[iNe].area] )
	node_feat1 = np.array(node_feat1)

	node_feat2 = []
	for l in  labels2Sub:
		iNe = iL2[l] # int(np.squeeze(np.argwhere(labels2==l)))	
		node_feat2.append([len(g2.edges(l)),rp2[iNe].area] )
	node_feat2 = np.array(node_feat2)

	gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=.1) # set affinity function
	K = pygm.utils.build_aff_mat(node_feat1, edgeFeatures1, conn1, node_feat2, edgeFeatures2, conn2, len(toAssigned1), None, len(toAssigned2), None, edge_aff_fn=my_edge_aff_fn, node_aff_fn=my_node_aff_fn)
	
	np.fill_diagonal(K, np.diagonal(K) + seed_subMat.T.reshape(-1) * K.max()*10)

	if showOn:
		#np.unravel_index(np.argmax(K),K.shape)
		plt.figure(figsize=(4, 4))
		plt.title(f'Affinity Matrix (size: {K.shape[0]}$\\times${K.shape[1]})')
		plt.imshow(seed_mat, cmap='Blues')
		plt.colorbar()
		plt.show()
		
		plt.figure(figsize=(4, 4))
		plt.title(f'Affinity Matrix (size: {K.shape[0]}$\\times${K.shape[1]})')
		plt.imshow(edgeFilter, cmap='Blues')#, vmax = K.max()/50)
		plt.colorbar()
		plt.show()

		#K[K<0.1]=0
		#np.fill_diagonal(K,0)
		plt.figure(figsize=(4, 4))
		plt.title(f'Affinity Matrix (size: {K.shape[0]}$\\times${K.shape[1]})')
		plt.imshow(K, cmap='Blues')#, vmax = K.max()/50)
		plt.colorbar()
		plt.show()

		valAffinity = np.concatenate(K)
		plt.hist(valAffinity[valAffinity!=0],bins=10)
		plt.show()

	X = pygm.rrwm(K, len(toAssigned1),len(toAssigned2),alpha = 0.8)
	if showOn:
		plt.figure(figsize=(8, 4))
		plt.subplot(1, 2, 1)
		plt.title('RRWM Soft Matching Matrix')
		plt.imshow(X, cmap='Blues', vmax = X.max()/50)
		plt.colorbar()
		plt.show()

	X = pygm.hungarian(X)
	if showOn:
		valFinal = np.concatenate(X)
		plt.hist(valAffinity[valAffinity!=0],bins=10)
		plt.show()

	if showOn:
		plt.figure(figsize=(8, 4))
		ax1 = plt.subplot(1, 2, 1)
		plt.title('Graph 1')
		nx.draw_networkx(g1Sub, pos=pos1)
		ax2 = plt.subplot(1, 2, 2)
		plt.title('Graph 2')
		nx.draw_networkx(g2Sub, pos=pos2)
		for i in range(len(X)):
			j = np.argmax(X[i]).item()
			if seed_mat[i, j]:
				line_color = "blue"
			#elif X_gt[i, j]:
			#    line_color = "green"
			else:
				line_color =  "red"
			con = ConnectionPatch(xyA=pos1[labels1Sub[i]], xyB=pos2[labels2Sub[j]], coordsA="data", coordsB="data",
					  axesA=ax1, axesB=ax2, color=line_color)
			plt.gca().add_artist(con)
		plt.show()


	plt.figure(figsize=(8, 4))
	ax1 = plt.subplot(1, 2, 1)
	plt.title('Graph 1')
	nx.draw_networkx(g1Sub, pos=pos1)
	ax2 = plt.subplot(1, 2, 2)
	plt.title('Graph 2')
	nx.draw_networkx(g2Sub, pos=pos2)
	matchNodes = np.argwhere(X==1)
	matchLabelSeg = np.zeros(imgSeg2.shape,dtype = np.int16)
	for mN in matchNodes:
		line_color = "blue"
		con = ConnectionPatch(xyA=pos1[labels1Sub[mN[0]]], xyB=pos2[labels2Sub[mN[1]]], coordsA="data", coordsB="data",
				  axesA=ax1, axesB=ax2, color=line_color)
		plt.gca().add_artist(con)
		xyz = np.argwhere(imgSeg2==labels2Sub[mN[1]])
		matchLabelSeg[xyz[:,0],xyz[:,1]] = labels1Sub[mN[0]]
	if showOn:
		plt.show()
	
	plt.close()
	if 1:#showOn:
		tifffile.imwrite(outImgFolder+"segMatch"+str(iteration)+".tif", matchLabelSeg) #ok
	
	assignedNodes1 = [labels1Sub[im] for im in matchNodes[:,0]]
	assignedNodes2 = [labels2Sub[im] for im in matchNodes[:,1]]
	
	iteration+=1
tifffile.imwrite(outImgFolder+"segMatch"+str(iteration)+".tif", matchLabelSeg) #ok
np.save(outImgFolder+'correspondingLabels.npy',[assignedNodes1,assignedNodes2]) #column 0 is before column 1 is after
#how to use : assignedNodes1,assignedNodes2 = np.load(outImgFolder+'correspondingLabels.npy')
####STOP iter 2
