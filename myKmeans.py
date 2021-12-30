#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 09:55:49 2020

@author: oehlers
"""
import matdf,clustering,func_collection
from sklearn.metrics import pairwise_distances_argmin_min, pairwise_distances
from sklearn.cluster import KMeans
from scipy.spatial import Voronoi
import numpy as np 
from copy import deepcopy
from scipy.spatial import ConvexHull

def get_kmeans(self,n_clusters,noTargetORwithTarget):
    assert isinstance(self,matdf.matdf), "Extension function for matdf.matdf only"
    dfset = deepcopy(self)
    norm_work,norm_test = dfset.get_normWork_normTest(noTargetORwithTarget)

    
    # there is no way of telling kmeans a min number of materials for each cluster!
    kmeans = KMeans(n_clusters=n_clusters).fit(norm_work)
    closest,_ = pairwise_distances_argmin_min(kmeans.cluster_centers_, norm_work)
    kmeans.norm_work = norm_work
    kmeans.norm_test = norm_test

    if noTargetORwithTarget in ["noTarget","withTarget","withTargetSqrt","withTargetEqual"]:
        test_labels = kmeans.predict(norm_test)
    elif noTargetORwithTarget=='withTargetProject': # exclude target for test label assignment as test points need to stay untouched!
        # added 25.01.2021
        test_labels, _ = pairwise_distances_argmin_min(norm_test[:, 1:], kmeans.cluster_centers_[:, 1:])

    distance_to_own_cluster_center_work = []
    distance_to_own_cluster_center_test = []
    for i in range(norm_work.shape[0]):
        distance_to_own_cluster_center_work += [pairwise_distances(norm_work[i,:].reshape(1, -1),
                                                                   kmeans.cluster_centers_[kmeans.labels_[i]].reshape(1, -1))[0,0]]
    for i in range(norm_test.shape[0]):
        target_exclusion = 1 if noTargetORwithTarget=='withTargetProject' else 0
        distance_to_own_cluster_center_test += [pairwise_distances(norm_test[i,target_exclusion:].reshape(1, -1),
                                                                   kmeans.cluster_centers_[test_labels[i],target_exclusion:].reshape(1, -1))[0,0]]
        
    max_distance_to_own_cluster_center_work = max(distance_to_own_cluster_center_work)
    max_distance_to_own_cluster_center_test = max(distance_to_own_cluster_center_test)
    
    probability_equivalent = np.array(distance_to_own_cluster_center_work)/max_distance_to_own_cluster_center_work
    strength_equivalent = np.array(distance_to_own_cluster_center_test)/max_distance_to_own_cluster_center_test
    
    workClusterDf = dfset.work
    testClusterDf = dfset.test
    
    # cluster center approach & save cluster materials:
    workClusterDf['labels'] = kmeans.labels_
    workClusterDf['probabilities'] = probability_equivalent
    
    testClusterDf['labels'] = test_labels
    testClusterDf['strengths'] = strength_equivalent
    
    kmeans.dfset = dfset
    
    kmeans.noTargetORwithTarget = noTargetORwithTarget

    kmeans.workClusterDf = workClusterDf
    kmeans.testClusterDf = testClusterDf

    kmeans.materialDict = {label: [list(workClusterDf[workClusterDf['labels']==label].index),
                                   list(testClusterDf[testClusterDf['labels']==label].index)]
                           for label in kmeans.labels_}
    assert np.sum([len(train_test_mat[0]) for train_test_mat in kmeans.materialDict.values()])==378 \
    and np.sum([len(train_test_mat[1]) for train_test_mat in kmeans.materialDict.values()])==504-378

    return kmeans

matdf.matdf.get_kmeans = get_kmeans

def assign_clusterCenters_as_trainpts(self,kmeans):
    assert isinstance(self,matdf.matdf), "Extension function for matdf.matdf only."
    assert isinstance(kmeans,KMeans), "Kmeans must be of type sklearn.cluster.Kmeans as obtained my myKmeans.get_kmeans only."
    dfset = self
    
    argmin_centers = pairwise_distances_argmin_min(kmeans.cluster_centers_, kmeans.norm_work)[0]
    
    trainmat = list(kmeans.dfset.work.iloc[argmin_centers,:].index)
    
    dfset.train = dfset.targeted.loc[trainmat,:]
    dfset.val = dfset.targeted
    
    return dfset

matdf.matdf.assign_clusterCenters_as_trainpts = assign_clusterCenters_as_trainpts
        