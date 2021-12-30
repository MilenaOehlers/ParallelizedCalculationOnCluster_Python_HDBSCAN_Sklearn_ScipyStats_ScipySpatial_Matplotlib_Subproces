#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 10:17:57 2020

@author: oehlers
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from copy import copy
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin_min
import pickle,json
import myfuncs
from scipy import stats


def dfisinstance(df):
    if isinstance(df, str):          returndf = pd.read_csv(df, sep="\s+", index_col=0).sort_index()
    if isinstance(df, pd.DataFrame): returndf = df.sort_index()
    return returndf

class matdf():
    def __init__(self, matdf, units, longname=None, explanation=None):
        """Creates whole and (default) targeted with matdf, and featuredf with units,longname,explanation"""
        self.distribution_enforced = False
        self.whole = self.targeted = dfisinstance(matdf)
        self.targets, self.features, self.train, self.work, self.val, self.test = None, None, None, None, None, None
        self.multi,self.cluster_dict,self.n_clusters = None,None,1
        self.metadata = ["targets","features","train.index","val.index","test.index"]
        
        self.Featuredf(units=units)
        self.Targets()
        
    def Featuredf(self,units,longname=None, explanation=None):
        assert not self.distribution_enforced, "Never use method EnforceTargetDistribution before this method!"

        self.featuredf = pd.DataFrame(index=self.whole.columns, data = {'usage': None, 'unit': units, 'name':longname, 'explanation':explanation})
        return self

    def Targets(self,use=0,drop=[]):
        """Creates targeted using whole by defining target property(ies) and discarding the rest"""
        assert not self.distribution_enforced, "Never use method EnforceTargetDistribution before this method!"

        targetdict = {'use': use, 'drop': drop}
        for useORdrop in targetdict.keys():
            if not isinstance(targetdict[useORdrop],list): targetdict[useORdrop] = [targetdict[useORdrop]]
            targetdict[useORdrop] = [numstr if isinstance(numstr,str) else self.whole.columns[numstr] for numstr in targetdict[useORdrop]]
        featurecols = [col for col in self.whole.columns if col not in targetdict['use']+targetdict['drop']]
        colseq = targetdict['use']+featurecols
        self.targeted = self.whole[colseq]
        self.targets = targetdict['use']
        self.features = featurecols
        self.featuredf['usage'] = ["target"   if self.featuredf.index[ind] in targetdict['use'] \
                              else "dropped"  if self.featuredf.index[ind] in targetdict['drop'] \
                              else "feature" for ind in range(len(self.featuredf.index))]
        return self

    def Test(self, ratio=0.25, seed=42):
        """Creates test and work by splitting targeted"""
        assert not self.distribution_enforced, "Never use method EnforceTargetDistribution before this method!"

        self.work, self.test = train_test_split(self.targeted, test_size=ratio, random_state=seed)
        self.test.ratio, self.test.seed = ratio, seed
        return self

    def Train(self, ratio, seed=13, subset=True):
        """Creates initial train and val, first by splitting work, second depending on subset by the split-rest or work"""
        assert not self.distribution_enforced, "Never use method EnforceTargetDistribution before this method!"

        if not hasattr(self, 'work'):
            self.work, self.test = self.targeted, None
        
        test_size = float(1.0-ratio) if isinstance(ratio,float) and ratio<=1 \
            else int(self.work.shape[0] - ratio) if isinstance(ratio,int) and ratio>=1 \
            else None
            
        if test_size==None: raise Exception("Ratio must be float between 0.0 and 1.0 or int > 1.")
        
        self.train, rest = train_test_split(self.work, test_size=test_size, random_state=seed) if ratio!=1 else (self.work, None)
        self.train.ratio, self.train.subset, self.train.seed= ratio, subset, seed
        
        self.val = self.work if subset is True else rest
        return self

    def MultiTask(self,pickled_cluster_dict_path):
        """Choose train and test set based on cluster results
        pickled_cluster_dict must have form { 1: [[a,b,c],[d,e,f]],2:...,...}
        where first list contains train, second test points for the given cluster label"""
        assert not self.distribution_enforced, "Never use method EnforceTargetDistribution before this method!"

        if isinstance(pickled_cluster_dict_path,str):
            with open(pickled_cluster_dict_path,'rb') as file:
                cluster_dict = pickle.load(file)
        else:
            cluster_dict = pickled_cluster_dict_path
            
        new_clusterdict = {}
        multi_traindf = pd.DataFrame()
        multi_testdf = pd.DataFrame()
        nsample_str = []
        tasknr = 1
        for key in sorted(list(cluster_dict.keys())):
            if not isinstance(key,int) or int(key)!=-1:
                trainlst, testlst = cluster_dict[key]
                if len(trainlst)>4 and len(testlst)>0:
                    this_traindf = self.targeted.loc[trainlst,:]
                    this_testdf = self.targeted.loc[testlst,:]
                    
                    this_traindf.index = [mat+"_task"+str(tasknr) for mat in trainlst]
                    this_testdf.index = [mat+"_task"+str(tasknr) for mat in testlst]
                    
                    new_clusterdict[tasknr] = cluster_dict[key]
                    
                    multi_traindf = multi_traindf.append(this_traindf)
                    multi_testdf = multi_testdf.append(this_testdf)
                    nsample_str += [str(len(trainlst))]
                    tasknr += 1
        self.train = multi_traindf
        self.val = multi_testdf
        self.test = multi_testdf
        self.train.index.name = 'materials'
        self.val.index.name = 'materials'
        self.test.index.name = 'materials'
        self.n_clusters = tasknr-1
        self.cluster_dict = new_clusterdict
        self.multi = ",".join(nsample_str)
        return self
    
    def SingleTask(self):
        assert not self.distribution_enforced, "Never use method EnforceTargetDistribution before this method!"
        self.train.index = [mat.split("_task")[0] for mat in self.train.index]
        self.val.index = [mat.split("_task")[0] for mat in self.val.index]
        self.test.index = [mat.split("_task")[0] for mat in self.test.index]
        self.train.index.name = 'materials'
        self.val.index.name = 'materials'
        self.test.index.name = 'materials'
        self.n_clusters = 1
        self.cluster_dict = None
        self.multi = None
        return self

    def EnforceTargetDistribution(self,apply=False,n=4):
        if apply:
            print("ENFORCING DISTRIBUTION________________________")
            assert hasattr(self,'train') and hasattr(self,'work') and hasattr(self,'test')
            self.distribution_enforced = True

            def get_repr_target(target_work_lst, n=4):
                assert n > 1, "Minimum two lattice constants!"
                sortedd = sorted(target_work_lst)
                ae, loce, scalee = stats.skewnorm.fit(sortedd)
                cdf = list(stats.skewnorm.cdf(sortedd, ae, loce, scalee))
                lenn = len(cdf)
                cdf_pos = [i / (n - 1) for i in range(n - 1)] + [1]
                idxs = [np.argmin([abs(c - pos) for c in cdf]) for pos in cdf_pos]
                x_vals = [sortedd[idx] for idx in idxs]
                return x_vals

            def insert_index_version(ind,v):
                if "_task" in ind:
                    mat,task = ind.split("_task")
                    newind = "{}v{}_task{}".format(mat,v,task)
                else:
                    newind = "{}v{}".format(ind,v)
                return ind #newind
            assert len(self.targets)==1, "EnforceDistribution method was written for one target only!"
            repr_target = get_repr_target(list(self.work.loc[:,self.targets[0]]),n)

            for v in range(n):
                train_slice = copy(self.train)
                train_slice.index = [insert_index_version(ind,v) for ind in self.work.index]

                if hasattr(self,'val') and self.val is not None:
                    val_slice = copy(self.val).loc[
                        [ind for ind in self.val.index
                         if ind not in list(self.train.index)+list(self.test.index)]
                    ,:]
                    if val_slice.shape[0]>0:
                        val_slice.index = [insert_index_version(ind,v) for ind in val_slice.index]
                        val_slice[self.targets] = [repr_target[v]]*val_slice.shape[0]

                test_slice = copy(self.test)
                test_slice.index = [insert_index_version(ind,v) for ind in self.test.index]
                test_slice[self.targets] = [repr_target[v]]*test_slice.shape[0]

                concat = [train_slice,val_slice,test_slice] \
                    if hasattr(self,'val') and self.val is not None and val_slice.shape[0]>0 \
                    else [train_slice,test_slice]
                new_train = pd.concat(concat,axis=0) if v==0 else pd.concat([new_train]+concat,axis=0)
                new_train = new_train.sort_index()
            assert new_train.shape[0] == self.targeted.shape[0]*n, "new_train_nrows!=targeted_nrows*n: {}!={}".format(
                new_train.shape[0],self.targeted.shape[0]*n
            )
            self.train = new_train
        else:
            print("____________________NOT ENFORCING DISTRIBUTION")
        return self

    def get_nsample(self):
        return self.multi if self.multi is not None else str(self.train.shape[0])

    def NewTrain(self,index):
        """Creates new train and val during iteration"""
        assert not self.distribution_enforced, "Never use method EnforceTargetDistribution before this method!"

        ratio, subset, seed = self.train.ratio, self.train.subset, None
        self.train = self.work.loc[index,:]
        self.val = self.work if subset is True else self.work.loc[[i for i in self.work.index if i not in self.train.index],:]
        self.train.ratio, self.train.subset, self.train.seed = ratio, subset, seed 

    def GreedyTrain(self,initial_trainsize,subset=False):
        assert not self.distribution_enforced, "Never use method EnforceTargetDistribution before this method!"

        norm_work = StandardScaler().fit_transform(self.work.values)
        kmeans = KMeans(n_clusters=initial_trainsize, random_state=0).fit(norm_work)
        closest,_ = pairwise_distances_argmin_min(kmeans.cluster_centers_, norm_work)
        self.train = self.work.iloc[closest,:]
        self.val = self.work if subset is True else self.work.loc[[i for i in self.work.index if i not in self.train.index],:]
        self.train.ratio, self.train.subset = float(initial_trainsize)/self.work.shape[0], subset
        return self
    

class results():
    def __init__(self,rmseORmaxae):
        testerr, ite_dim = 'test-{}'.format(rmseORmaxae), pd.MultiIndex.from_product([[],[]], names=['ite','dim'])
        
        self.errtype, self._proxy, self.ite, self.limes = rmseORmaxae, None, -1, '-'
        
        self.allmodels = pd.DataFrame(index=ite_dim, columns=['chosen','trainmat','change','descs','coefs','interc','train-rmse','train-maxae','val-rmse','val-maxae','test-rmse','test-maxae','rank'])
        self.sisspace  = {(ite,valORtest): pd.DataFrame(index='materials', columns=['y','pred','error','desc1-...']) for ite in [] for valORtest in []}
        self.bestmodel = {key: None for key in ['bestite','dim','trainmat','descs','coefs','interc',testerr,testerr+'-toinitial',testerr+'-towork']}
        
    def proxy(self,proxysis):
        psis, testerr = proxysis.pred, 'test-{}'.format(self.errtype)
        if psis.success == 1: 
            self._proxy = float(copy(psis.outdf[psis.outdf['chosen']==1][testerr])) 
        return self
    
    def add(self,sisso,meth_success=1):
        if sisso.pred.success == 1 and meth_success == 1: 
            self.ite += 1
            
            psis, ite = sisso.pred, self.ite 
            fromoutdf = sorted(set(self.allmodels.columns)-{'trainmat','change'})
            
            for dim in psis.outdf.index:
                self.allmodels.loc[(ite,dim),'trainmat'] = [matnow] = [set(copy(sisso.fit.matdf.train.index))]
                self.allmodels.loc[(ite,dim),'change'] = len(matnow - self.allmodels.loc[(ite-1,1),'trainmat'])/len(matnow) if ite!=0 else 1
                for col in fromoutdf: self.allmodels.loc[(ite,dim),col] = copy(psis.outdf.loc[dim, col])
                if psis.outdf.loc[dim,'chosen'] == 1: 
                    self.sisspace[(ite,'val')] = copy(psis.valspace[dim])
                    self.sisspace[(ite,'test')] = copy(psis.testspace[dim])
                    
        else: 
            self.limes = 'fail'
            
        return self
    
    def ana(self,maxlooplen):
        chosenres, ite  = self.allmodels[self.allmodels['chosen']==1].reset_index('dim'), self.ite
        testerr, valerr = 'test-{}'.format(self.errtype), 'val-{}'.format(self.errtype)
        
        for i in range(1,ite):
            if chosenres.loc[ite,'trainmat'] == chosenres.loc[ite-i,'trainmat'] and self.limes=='-': 
                self.limes = 'conv'                if i==1 else i 
                self.bestmodel['bestite']=bite=ite if i==1 else ite-i+np.argmin(list(chosenres.loc[ite-i:ite,valerr]))
                self.bestmodel.update({col: chosenres.loc[bite,col] for col in ['dim','trainmat','descs','coefs','interc',testerr]})
                self.bestmodel[testerr+'-toinitial'] = float(chosenres.loc[bite,testerr])/float(chosenres.loc[0,testerr])
                if self._proxy!=None: 
                    self.bestmodel[testerr+'-towork'] = float(chosenres.loc[bite,testerr])/float(self._proxy)
                
