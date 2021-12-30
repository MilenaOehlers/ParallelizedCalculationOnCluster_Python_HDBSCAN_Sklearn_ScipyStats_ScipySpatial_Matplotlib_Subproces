#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 09:32:06 2020

@author: oehlers
"""
import matdf,sklearn,hdbscan,random,pickle,os,copy,mysis,extensions
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import machinejob as mj
import sklearn
import pandas as pd
from copy import copy

def get_perov_standard():
    return get_dfset_for_sisso("/home/oehlers/Documents/masterthesis/02-data/data/cubic_perovskites.dat",
                               ['t1','t2','t3','t4']+['a']*4+['b','b']+['c']*10,
                               [['lat'],['bul','o_hirsh','o_center']],
                               "/home/oehlers/Documents/masterthesis/02-data/data/cubic_perovskites_testlst")

def get_dfset_for_sisso(datapath,units,targetlst,testlstpath):
    if isinstance(testlstpath,str):
        with open(testlstpath,"rb") as file:
            testlst = pickle.load(file)
    dfset = matdf.matdf(datapath, units=units).Targets(*targetlst)
    dfset.test = dfset.targeted.loc[testlst,:]
    dfset.work = dfset.targeted.loc[[mat for mat in list(dfset.targeted.index) if mat not in testlst],:]
    dfset.dataset = datapath.split("/")[-1].split('.')[0]
    dfset.testlst = testlstpath.split("/")[-1]
    return dfset

def get_dfset_for_clustering_in_primary_space(p):
    clustering_dfset = get_dfset_for_sisso(p.datafile, p.units, p.targetlst, p.testlstfile)
    clustering_dfset.train = copy(clustering_dfset.work)
    clustering_dfset.val = copy(clustering_dfset.test)
    # print(dfset.targeted.head())
    if 'withTarget' in p.method:
        get_testtargets_sis = mysis.sisso(clustering_dfset, p.desc_dim, p.rung, p.rmseORmaxae)
        for mat in clustering_dfset.test.index:
            clustering_dfset.test.loc[mat, clustering_dfset.targets] = get_testtargets_sis.pred.testspace[3].loc[
                mat, 'pred']
    return clustering_dfset

def get_dfset_for_clustering_in_3d_sisso_space(p,coefs_stretch=True):
    df_name = "{}_sisspace_{}_{}".format(p.dataset,p.target,p.test_shuffle)
    if df_name not in os.listdir(p.datapath):
        print("df here not calc yet ")
        dfset_for_1st_sisso = get_dfset_for_sisso(p.datafile, p.units, p.targetlst, p.testlstfile)
        dfset_for_1st_sisso.train = copy(dfset_for_1st_sisso.work)
        dfset_for_1st_sisso.val = copy(dfset_for_1st_sisso.work)
        sis1 = mysis.sisso(dfset_for_1st_sisso, p.desc_dim, p.rung, p.rmseORmaxae)

        valsp = sis1.pred.valspace[3].drop(columns=['pred', 'error'])
        testsp = sis1.pred.testspace[3].drop(columns=['y', 'error']).rename(columns={'pred': 'y'})
        if coefs_stretch:
            for d in range(3):
                valsp.loc[:,'desc{}'.format(str(d+1))] = [float(sis1.pred.outdf.loc[3,'coefs'][0][d])
                                                         * float(i) for i in valsp.loc[:,'desc{}'.format(str(d+1))]]
                testsp.loc[:, 'desc{}'.format(str(d + 1))] = [float(sis1.pred.outdf.loc[3,'coefs'][0][d])
                                                         * float(i) for i in testsp.loc[:,'desc{}'.format(str(d+1))]]
        else:
            pass
        whole = pd.concat([valsp, testsp])
        whole.index.name = 'materials'
        whole.to_csv(os.path.join(p.datapath,df_name),sep=',',index=True)
        print('saved df')
    else:
        print('df calced')
        whole = pd.read_csv(os.path.join(p.datapath,df_name),sep=',',index_col='materials')
        testlst = pickle.load(open(os.path.join(p.datapath,p.dataset)+"_testlst",'rb'))
        testsp = whole.loc[testlst,:]
        valsp = whole.loc[[mat for mat in list(whole.index) if mat not in testlst],:]
    dfset_for_clustering_in_3d_sisso_space = matdf.matdf(whole, ['t', 'a', 'b', 'c'])
    dfset_for_clustering_in_3d_sisso_space.train, dfset_for_clustering_in_3d_sisso_space.val, dfset_for_clustering_in_3d_sisso_space.work = valsp, valsp, valsp
    dfset_for_clustering_in_3d_sisso_space.test = testsp
    dfset_for_clustering_in_3d_sisso_space.dataset = p.datafile.split("/")[-1].split('.')[0]
    dfset_for_clustering_in_3d_sisso_space.testlst = p.testlstfile.split("/")[-1]
    return dfset_for_clustering_in_3d_sisso_space

def get_normWork_normTest(self,noTargetORwithTarget):
    assert isinstance(self,matdf.matdf), "Extension function for matdf.matdf only"
    dfset = self
    if noTargetORwithTarget=="noTarget":
        workClusterDf = dfset.work.drop(dfset.targets,1)
        testClusterDf = dfset.test.drop(dfset.targets,1)
        
        norm_work = StandardScaler().fit_transform(workClusterDf.values)
        norm_test = StandardScaler().fit_transform(testClusterDf.values)
    elif "withTarget" in noTargetORwithTarget:
        workClusterDf = dfset.work
        testClusterDf = dfset.test
        n_feats = workClusterDf.shape[1]-1
        
        # to mean0, std1 normalized dataset: features are divided by #of features iot always weigh target equally
        norm_work = StandardScaler().fit_transform(workClusterDf.values)
        norm_test = StandardScaler().fit_transform(dfset.test.values)
        for i in range(norm_work.shape[0]):
            for j in range(norm_work.shape[1]):
                if j!=0:
                    if noTargetORwithTarget=='withTarget': norm_work[i,j] = norm_work[i,j]/(n_feats)
                    if noTargetORwithTarget=='withTargetSqrt': norm_work[i,j] = norm_work[i,j]/(n_feats)**0.5
                    if noTargetORwithTarget=='withTargetEqual': norm_work[i,j] = norm_work[i,j]
        for i in range(norm_test.shape[0]):
            for j in range(norm_test.shape[1]):
                if j!=0:
                    if noTargetORwithTarget=='withTarget': norm_test[i,j] = norm_test[i,j]/(n_feats)
                    if noTargetORwithTarget=='withTargetSqrt': norm_test[i,j] = norm_test[i,j]/(n_feats)**0.5
                    if noTargetORwithTarget=='withTargetEqual': norm_test[i,j] = norm_test[i,j]
    else:
        raise Exception("noTargetORwithTarget must be 'noTarget' or contain 'withTarget'")
    return norm_work,norm_test

matdf.matdf.get_normWork_normTest = get_normWork_normTest

def get_clusterMaterialsDict(self):
    
    if not hasattr(self,"clusterMaterialsDict"):
            
        assert isinstance(self,hdbscan.HDBSCAN) or isinstance(self,sklearn.cluster.KMeans) \
                or isinstance(self,sklearn.cluster.DBSCAN), "Extension function for hdbscan.HDBSCAN as obtained by matdf.matdf.get_clusterer OR sklearn.cluster.KMeans as obtained by matdf.matdf.get_keans only."
        
        workClusterDf = self.workClusterDf
        testClusterDf = self.testClusterDf
        clusterMaterialsDict = {}
        set_labels = sorted(list(set(list(workClusterDf['labels']))-set([-1])))
        for label in set_labels:
            label_selection = workClusterDf[workClusterDf['labels']==label]
            test_label_selection = testClusterDf[testClusterDf['labels']==label]
            #centerlst += [label_selection['probabilities'].idxmax()]
            clusterMaterialsDict[label] = [list(label_selection.index),list(test_label_selection.index)]
        self.clusterMaterialsDict = clusterMaterialsDict
        
    return self.clusterMaterialsDict
    
hdbscan.HDBSCAN.get_clusterMaterialsDict = get_clusterMaterialsDict
sklearn.cluster.KMeans.get_clusterMaterialsDict = get_clusterMaterialsDict
sklearn.cluster.DBSCAN.get_clusterMaterialsDict = get_clusterMaterialsDict

def pick_randomly_uniform_among_clusters(self, clustererORclusterDict, npicked_per_cluster):
    assert isinstance(self, matdf.matdf), "Extension function for matdf.matdf only."
    assert isinstance(clustererORclusterDict, hdbscan.HDBSCAN) \
           or isinstance(clustererORclusterDict, sklearn.cluster.KMeans) \
           or isinstance(clustererORclusterDict,dict)
    dfset = self

    if isinstance(clustererORclusterDict,(hdbscan.HDBSCAN,sklearn.cluster.KMeans)):
        set_labels = list(set(list(clustererORclusterDict.workClusterDf['labels'])) - {-1})
        matlst = []
        for label in set_labels:
            allmat = list(clustererORclusterDict.workClusterDf[
                              clustererORclusterDict.workClusterDf['labels'] == label].index)
            random.shuffle(allmat)
            matlst += allmat[:npicked_per_cluster]
    else:
        set_labels = list(set(list(clustererORclusterDict.keys())) - {-1})
        matlst = []
        for label in set_labels:
            assert len(clustererORclusterDict[label])==2 and all(isinstance(clustererORclusterDict[label][i],list)
                                                                 for i in [0,1])
            allmat = clustererORclusterDict[label][0]
            random.shuffle(allmat)
            matlst += allmat[:npicked_per_cluster]

    dfset.train = dfset.targeted.loc[matlst, :]
    dfset.val = dfset.work.loc[[mat for mat in dfset.work.index if mat not in matlst], :]
    assert len(dfset.train.index)==npicked_per_cluster*len(set_labels)
    return dfset

matdf.matdf.pick_randomly_uniform_among_clusters = pick_randomly_uniform_among_clusters

def assign_trainpts_randomly(self,clustererORclusterMasterialsDict,trainr):
    assert isinstance(self,matdf.matdf), "Extension function for matdf.matdf only."
    assert isinstance(clustererORclusterMasterialsDict,hdbscan.HDBSCAN) \
            or isinstance(clustererORclusterMasterialsDict,sklearn.cluster.KMeans) \
            or isinstance(clustererORclusterMasterialsDict,dict), "clustererORclusterMasterialsDict must be hdbscan.HDBSCAN or sklearn.cluster.KMeans as obtained by resp. get-methods in clustering-package OR must be a clusterMaterialsDict."
    
    dfset = self
    
    if isinstance(clustererORclusterMasterialsDict,dict):
        matlst = []
        for clusternr, train_test_clusterMaterials in clustererORclusterMasterialsDict.items():
            allmat = train_test_clusterMaterials[0]
            random.shuffle(allmat)
            matlst += allmat[:max(1,int(len(allmat)*trainr))]
    else:
        set_labels = list(set(list(clustererORclusterMasterialsDict.workClusterDf['labels']))-set([-1]))
        matlst = []
        for label in set_labels:
            allmat = list(clustererORclusterMasterialsDict.workClusterDf[clustererORclusterMasterialsDict.workClusterDf['labels']==label].index)
            random.shuffle(allmat)
            matlst += allmat[:max(1,int(len(allmat)*trainr))]
         
    dfset.train = dfset.targeted.loc[matlst,:]
    dfset.val = dfset.work.loc[[mat for mat in dfset.work.index if mat not in matlst],:]
    
    return dfset

matdf.matdf.assign_trainpts_randomly = assign_trainpts_randomly

def get_merged_clusterMaterialsDict(self):
    
    clusterer = self
    
    dicts_to_merge = [attr for attr in ['clusterMaterialsDict','offsplitMaterialsDicts','offsplitMaterialsDicts_forMultiSISSO'] 
    if hasattr(self,attr) and ( not hasattr(self,attr+"_is_merged") or getattr(self,attr+"_is_merged") is False ) ]
    
    for dict_to_merge in dicts_to_merge:
        
        setattr(clusterer,dict_to_merge+"_is_merged",True)
        
        matDict = getattr(clusterer,dict_to_merge)
        
        if isinstance( matDict[list(matDict.keys())[0]], dict ):
            for n_offsplit in matDict.keys():
                matDict[n_offsplit] = old_to_merged_matDict(matDict[n_offsplit],dict_to_merge)
        else:
            matDict = old_to_merged_matDict(matDict,dict_to_merge)
        
        setattr( clusterer, dict_to_merge, matDict )
        self = clusterer
    
    return self
    
hdbscan.HDBSCAN.get_merged_clusterMaterialsDict = get_merged_clusterMaterialsDict
sklearn.cluster.KMeans.get_merged_clusterMaterialsDict = get_merged_clusterMaterialsDict