#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 13:16:01 2020

@author: oehlers
"""
# IGNORE UNUSED WARNING for extensions & clustering! as extension functions, packages never explicitly called but still does add methods to existing classes (like sort() to pd.DataFrame)
import matdf,os,hdbscan,extensions,clustering,sklearn
import pandas as pd
import numpy as np
from copy import deepcopy,copy
import pickle


def get_clusterer(self,noTargetORwithTarget,min_cluster_size=None,save_at=None):
    dfset = self
    
    filename = "hdbscan_clusterer_{}_{}_{}_{}".format(
                noTargetORwithTarget,min_cluster_size,self.dataset,self.testlst)
    
    if save_at is not None and filename in os.listdir(save_at):
        with open(os.path.join(save_at,filename),'rb') as f:
            clusterer =  pickle.load(f)
        print("LOADED")
            
    else:
        assert isinstance(dfset,matdf.matdf), "Extension function for matdf.matdf only."
        # hdbscan predict cluster membership of new points: 
        # https://hdbscan.readthedocs.io/en/latest/prediction_tutorial.html
        # clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True, prediction_data=True)
        # clusterer.fit(test_data)
        # test_labels, strengths = hdbscan.approximate_predict(clusterer, test_points)
        if min_cluster_size is None:
            min_cluster_size = 5
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, gen_min_span_tree=True, prediction_data=True)

        norm_work,norm_test = dfset.get_normWork_normTest(noTargetORwithTarget)
        
        clusterer.fit(norm_work)
        
        workClusterDf = dfset.work
        testClusterDf = dfset.test
        # cluster center approach & save cluster materials:
        workClusterDf['labels'] = clusterer.labels_
        workClusterDf['probabilities'] = clusterer.probabilities_
        
        test_labels,strengths = hdbscan.prediction.approximate_predict(clusterer, norm_test)
        
        testClusterDf['labels'] = test_labels
        testClusterDf['strengths'] = strengths
        
        clusterer.dfset = dfset
        
        clusterer.noTargetORwithTarget = noTargetORwithTarget
        clusterer.min_cluster_size = min_cluster_size
        clusterer.norm_work = norm_work
        clusterer.norm_test = norm_test
        
        clusterer.workClusterDf = workClusterDf
        clusterer.testClusterDf = testClusterDf
        
        clusterer.save(save_at)
    
    return clusterer

matdf.matdf.get_clusterer = get_clusterer


def save(self, path):
    if path is not None:
        filename = "hdbscan_clusterer_{}_{}_{}_{}".format(
            self.noTargetORwithTarget, self.min_cluster_size, self.dfset.dataset, self.dfset.testlst)

        with open(os.path.join(path, filename), 'wb') as f:
            pickle.dump(self, f)


hdbscan.HDBSCAN.save = save


#region:new cluster_split_functions:
def get_clusters(lambda_val, tree_df, workdf, printon=False, printheritance=True):
    tree_df = tree_df.sort_values(by=['lambda_val'], ascending=True)

    cluster_splits = tree_df[tree_df['child_size'] != 1]
    cluster_members = tree_df[tree_df['child_size'] == 1]

    disconnected = tree_df[tree_df['lambda_val'] <= lambda_val]
    connected = tree_df[tree_df['lambda_val'] > lambda_val]

    outliers = [workdf.index[child_ind] for child_ind in disconnected[disconnected['child_size'] == 1]['child']]
    leaf_clusters_df = connected[connected['child_size'] == 1]
    leaf_clusters_dict = {int(parent): [workdf.index[child_ind] for child_ind in
                                        leaf_clusters_df[leaf_clusters_df['parent'] == parent]['child']]
                          for parent in set(leaf_clusters_df['parent'])}
    connected_clusters = connected[connected['child_size'] != 1].sort_values(by=['child'], ascending=False)
    cluster_heritance = {}
    for ind in connected_clusters.index:
        parent, child, _lam, child_size = connected_clusters.loc[ind, :]
        if parent not in cluster_heritance:
            cluster_heritance[int(parent)] = []
        if child not in list(leaf_clusters_dict.keys()) + list(cluster_heritance.keys()):
            raise Exception("Child is neither in leafs nor cluster heritance.")
        if child in leaf_clusters_dict.keys():
            cluster_heritance[int(parent)] += leaf_clusters_dict[int(child)]
            del leaf_clusters_dict[child]
        if child in cluster_heritance.keys():
            cluster_heritance[int(parent)] += cluster_heritance[int(child)]
            del cluster_heritance[int(child)]

    all_clusters = copy(cluster_heritance)

    for leafkey in leaf_clusters_dict.keys():
        if leafkey in all_clusters.keys():
            all_clusters[leafkey] += leaf_clusters_dict[leafkey]
        else:
            all_clusters[leafkey] = leaf_clusters_dict[leafkey]
    all_clusters.update({-1: outliers})
    splits_into = {tuple(set(cluster_splits[cluster_splits['lambda_val'] == lambda_val]['parent'])):
                       tuple(set(cluster_splits[cluster_splits['lambda_val'] == lambda_val]['child']))}
    return all_clusters, splits_into

def get_lambda2clusters_lambda2splits(tree_df,workdf):
    cluster_splits = tree_df[tree_df['child_size'] != 1]
    cluster_members = tree_df[tree_df['child_size'] == 1]
    return {lam: get_clusters(lam,tree_df,workdf)[0] for lam in sorted(list(set(cluster_splits['lambda_val'])))},\
            {lam: get_clusters(lam,tree_df,workdf)[1] for lam in sorted(list(set(cluster_splits['lambda_val'])))}


def get_lambda2offsplits(tree_df, workdf):
    """
    With each lambda step, one cluster is added.
    ( former Exception: Lambda step splits min cluster size or more
    points off at once, that are not connected among each other due to same mutual distance as offsplit.
    These are listed in condensed tree as clusters then, but really arent.
    -> these former exceptions are deleted now iot get right presentation )
    """
    lambda2clusters, lambda2splits = get_lambda2clusters_lambda2splits(tree_df, workdf)
    lambda2offsplits = {lam: clust for lam, clust in [sorted(list(lambda2clusters.items()))[0]]}
    for lambda_val in sorted(lambda2clusters.keys())[1:]:
        # get offsplits for previous lambda:
        previous_lambda_val = list(lambda2clusters.keys())[list(lambda2clusters.keys()).index(lambda_val) - 1]
        previous_offsplits = copy(lambda2offsplits[previous_lambda_val])
        parents, childs = list(lambda2splits[lambda_val].items())[0]
        # & delete cluster thats split in two and outliers:
        for parent in list(parents) + [-1]:
            del previous_offsplits[parent]
            previous_offsplits.update({clusternr: cluster
                                       for clusternr, cluster in lambda2clusters[lambda_val].items()
                                       if clusternr in childs})
        # -> new cluster offsplits:
        resulting_offsplits = copy(previous_offsplits)
        # only take over those outliers that are not inside any of the final clusters
        non_outliers = []
        for clusternr, cluster in resulting_offsplits.items():
            non_outliers += cluster
        possible_outliers = lambda2clusters[lambda_val][-1]
        true_outliers = sorted(list(set(possible_outliers) - set(non_outliers)))
        # save resulting new offsplit:
        resulting_offsplits.update({-1: true_outliers})
        lambda2offsplits.update({lambda_val: resulting_offsplits})

        former_lambda = None
        lambdas_to_delete = []

    for lam, offsplits in lambda2offsplits.items():
        if former_lambda is not None:
            if len(offsplits) == len(lambda2offsplits[former_lambda]):
                lambdas_to_delete += [lam]
        former_lambda = lam
    for lam in lambdas_to_delete:
        del lambda2offsplits[lam]

    return lambda2offsplits, lambda2splits


def get_offsplitMaterialsDicts_forMultiSISSO(self, rigid_lambda=False):
    """rigid_lambda==True:  Returns clusters as can be seen in dendogram for fixed lambda. Advantage: Closer to
                               original hdbscan method.
                     False: Returns clusters that split off before regarded lambda value, before losing points
                               which are discarded for outliers. Thus for each 'leaf', its offsplit lambda is used.
                               Advantage: Less outliers, for each offsplit number of clusters increased by one.
    Test pt assignment: Mutual distance calculated as if all positions of train and test materials had been known
                        from the start.
                        For test materials, only mutual distances are considered of points that are not outliers."""

    clusterer = self
    workdf = self.dfset.work
    testdf = self.dfset.test
    tree_df = clusterer.condensed_tree_.to_pandas()

    if rigid_lambda:
        lambda2clusters, _ = get_lambda2clusters_lambda2splits(tree_df, workdf)
    else:
        lambda2clusters, _ = get_lambda2offsplits(tree_df, workdf)

    # rename clusternumbers ([-1]:outliers, [>0]:clusters):
    for lambda_val, clusters in lambda2clusters.items():
        clusters = {i if sorted(list(clusters.keys()))[i] != -1 else -1
                    : clusters[sorted(list(clusters.keys()))[i]]
                    for i in range(len(sorted(list(clusters.keys()))))}
        lambda2clusters[lambda_val] = clusters

    ## assign test points

    # preparation calculation mutual distances:
    all_trainmat = list(workdf.index)
    all_testmat = list(testdf.index)

    if self.noTargetORwithTarget == "noTarget":   nocols = ["labels", "probabilities"] + self.dfset.targets
    if self.noTargetORwithTarget == "withTarget": nocols = ["labels", "probabilities"]
    columns = [col + "_normed" for col in self.dfset.work.columns if col not in nocols]

    norm_work_df = pd.DataFrame(self.norm_work, index=self.dfset.work.index, columns=columns)
    norm_test_df = pd.DataFrame(self.norm_test, index=self.dfset.test.index, columns=columns)

    all_trainmat = list(norm_work_df.index)
    all_testmat = list(norm_test_df.index)

    data1 = [[np.linalg.norm(np.array(norm_work_df.loc[trainmat, columns])
                             - np.array(norm_work_df.loc[mat, columns])) for mat in all_trainmat]
             for trainmat in all_trainmat]
    data2 = [[np.linalg.norm(np.array(norm_work_df.loc[trainmat, columns])
                             - np.array(norm_test_df.loc[mat, columns])) for mat in all_testmat]
             for trainmat in all_trainmat]
    eucl_distance_matrix_train = pd.DataFrame(data1, index=all_trainmat, columns=all_trainmat)
    eucl_distance_matrix_test = pd.DataFrame(data2, index=all_trainmat, columns=all_testmat)

    core_dist = {}
    for mat in all_testmat:
        eucl_dist = deepcopy(eucl_distance_matrix_test)
        core_dist[mat] = eucl_dist[mat].sort_values(ascending=True)[4]
    for mat in all_trainmat:
        eucl_dist_train = deepcopy(eucl_distance_matrix_train)
        eucl_dist_test = deepcopy(eucl_distance_matrix_test)
        traindists = list(eucl_dist_train[mat].sort_values(ascending=True))[1:6]
        mintestdist = eucl_dist_test.loc[mat, :].sort_values(ascending=True)[0]
        core_dist[mat] = sorted(list(traindists) + [mintestdist])[4]

    # test point assignment based on mutual distances and respective clustering,
    # only assignment to non-outlier train points possible
    lambda2clustersAlsoTestpts = {}
    for lam in lambda2clusters.keys():
        clusters = lambda2clusters[lam]
        trainpt2clusternr = {trainpt: clusternr for clusternr in clusters.keys() for trainpt in clusters[clusternr]}

        trainmat_no_outliers = []
        for clusternr in list(set(clusters.keys()) - set([-1])):
            trainmat_no_outliers += clusters[clusternr]
        mutual = [[max(eucl_distance_matrix_test.loc[train, test], core_dist[train], core_dist[test])
                   for test in all_testmat]
                  for train in trainmat_no_outliers]
        mutual_distances = pd.DataFrame(mutual, index=trainmat_no_outliers, columns=all_testmat)

        closest_trainpt_no_outlier = mutual_distances.idxmin()

        closest = pd.DataFrame(index=all_testmat, columns=['closest_trainpt_no_outlier', 'lambda'])
        closest['closest_trainpt_no_outlier'] = list(closest_trainpt_no_outlier)
        closest['lambda'] = [1 / min(mutual_distances[mat]) for mat in all_testmat]
        closest['outlier'] = [closestlam < lam for closestlam in closest['lambda']]
        closest['clusternr'] = [trainpt2clusternr[trainpt] if not outlier else -1
                                for trainpt, outlier in zip(list(closest['closest_trainpt_no_outlier']),
                                                            list(closest['outlier']))]
        closest.index.name = 'testpt'

        clusters_with_testpts = {clusternr: [clusters[clusternr],
                                             list(closest[closest['clusternr'] == clusternr].index)]
                                 for clusternr in clusters.keys()}
        lambda2clustersAlsoTestpts[lam] = clusters_with_testpts

    self.offsplitMaterialsDicts_forMultiSISSO = lambda2clustersAlsoTestpts
    return lambda2clustersAlsoTestpts

hdbscan.HDBSCAN.get_offsplitMaterialsDicts_forMultiSISSO = get_offsplitMaterialsDicts_forMultiSISSO

def get_nclusters2clusterDicts(self):
    if not hasattr(self,"offsplitMaterialsDicts_forMultiSISSO"):
        self.get_offsplitMaterialsDicts_forMultiSISSO(rigid_lambda=False)
    lambda2clusterDicts = self.offsplitMaterialsDicts_forMultiSISSO

    nclusters2clusterDicts = {}
    for lam,clusterDict in sorted(lambda2clusterDicts.items()):
        nclust = len(set(clusterDict.keys())-{-1})
        nclusters2clusterDicts[nclust] = clusterDict
    return nclusters2clusterDicts

hdbscan.HDBSCAN.get_nclusters2clusterDicts = get_nclusters2clusterDicts

def reassignLabels_deleteProb_for_offsplitIte(self,ite):
    
    assert isinstance(self,hdbscan.HDBSCAN)
    assert hasattr(self,"offsplitMaterialsDicts")
    assert ite in self.offsplitMaterialsDicts.keys()
    
    if 'probabilities' in self.workClusterDf.columns: 
        self.workClusterDf = self.workClusterDf.drop('probabilities',1)
        self.testClusterDf = self.testClusterDf.drop("strengths",1).drop("labels",1)
    
    matdict = self.offsplitMaterialsDicts[ite]
    
    for clusternr,materials in matdict.items():
        self.workClusterDf.loc[materials[0],"labels"] = int(clusternr)
    return self

hdbscan.HDBSCAN.reassignLabels_deleteProb_for_offsplitIte = reassignLabels_deleteProb_for_offsplitIte


   
 