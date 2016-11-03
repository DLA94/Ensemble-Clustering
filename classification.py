import numpy as np
import matplotlib.pyplot as plt

from itertools import cycle
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.datasets import make_blobs
from ensemble_clustering import relabel_cluster, voting

def loadDataSet(fileName):
	fr = open(fileName)
	X = []
	y = []
	i = 0
	for line in fr.readlines():
		curLine = line.strip().replace(' ','').split(',')
		if i == 0:
			i = 1
			features_name = curLine[1:-1]
		else:
			xtemp = curLine[1:-1]
			X.append(xtemp)
			y.append(curLine[-1])
	return X, y
	
def Generate_Random_Sample():
	centers = [[1, 1], [-1, -1]]
	X, y = make_blobs(n_samples = 1000, centers = centers, cluster_std = 0.6)
	return X, y

def Test_Accuaracy(y, y_pred):
	wrong_number = 0
	tmp_relabel = [y, y_pred]
	tmp_relabel = relabel_cluster(tmp_relabel)
	y = tmp_relabel[0]
	y_pred = tmp_relabel[1]
	for index in range(len(y)):
		if y[index] != y_pred[index]:
			wrong_number += 1
	return format(((len(y) * 1.0 - wrong_number) / len(y)), ".8%")

def Clustering_Ensemble(X, y, n_clusters):
	y_KMeans = KMeans(n_clusters = n_clusters).fit(X)
	# print y_KMeans.labels_
	print Test_Accuaracy(y, y_KMeans.labels_)
	y_SC = SpectralClustering(n_clusters = n_clusters).fit(X)
	# print y_SC.labels_
	print Test_Accuaracy(y, y_SC.labels_)
	y_AC = AgglomerativeClustering(n_clusters = n_clusters).fit(X)
	# print y_AC.labels_
	print Test_Accuaracy(y, y_AC.labels_)
	clusters = []
	clusters.append(list(y_KMeans.labels_))
	clusters.append(list(y_SC.labels_))
	clusters.append(list(y_AC.labels_))
	# print "=========="
	# for cluster in clusters:
	# 	print cluster
	relabeled_clusters = relabel_cluster(clusters)
	# print "=========="
	# for cluster in relabeled_clusters:
	# 	print cluster
	print "=========="
	y_pred = voting(relabeled_clusters)
	print Test_Accuaracy(y, y_pred)
	# print y_pred
	return np.array(X), y_pred
	

def Plot_Result(X ,y_pred):
	plt.close('all')
	plt.figure()
	plt.clf()
	
	colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
	colors = np.hstack([colors] * 20)
	
	plt.scatter(X[:, 0], X[:, 1], color=colors[y_pred].tolist())
	plt.show()

def main():
	X, y = loadDataSet('julei.csv')
	# X, y = Generate_Random_Sample()
	X, y_pred = Clustering_Ensemble(X, y, 2)
	Plot_Result(X, y_pred)


if __name__ == '__main__':
	main()