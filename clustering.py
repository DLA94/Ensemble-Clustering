import numpy as np
import matplotlib as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from ensemble_clustering import relabel_cluster, voting

def load_dataset(fileName):
	""" load the dataset from csv file """
	fr = open(fileName)
	# X is attribute
	# y is target
	X = []
	y = []
	i = 0
	for line in fr.readlines():
		curLine = line.strip().replace(' ','').split(',')
		if i == 0:
			# the first line is the features_name
			i = 1
			features_name = curLine[1:-1]
		else:
			xtemp = curLine[1:-1]
			X.append(xtemp)
			y.append(curLine[-1])
	return X, y

def ensemble_clusters(X, y):
	# KMeans Algorithm
	y_KMeans = KMeans(n_clusters = 2).fit(X)
	# print y_KMeans.labels_
	
	# SpectralClustering Algorithm
	y_SC = SpectralClustering(n_clusters = 2).fit(X)
	# print y_SC.labels_
	
	# AgglomerativeClustering Algorithm
	y_AC = AgglomerativeClustering(n_clusters = 2).fit(X)
	# print y_AC.labels_
	
	# ensembling phase
	clusters = []
	clusters.append(list(y_KMeans.labels_))
	clusters.append(list(y_SC.labels_))
	clusters.append(list(y_AC.labels_))
	print "=========="
	for cluster in clusters:
		print cluster
	print "=========="
	relabeled_clusters = relabel_cluster(clusters)
	for cluster in relabeled_clusters:
		print cluster
	print "=========="
	print voting(relabeled_clusters)

if __name__ == '__main__':
	X, y = load_dataset('julei.csv')
	ensemble_clusters(X, y)