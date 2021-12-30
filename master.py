#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 10:37:58 2020
@author: oehlers
"""    
class params():
    def __init__(self):
        from datetime import datetime
        import machinejob as mj
        import os
        from pathlib import Path
        
        self.testvar = 1
        self.dataset = 'cubic_perovskites'
        self.target = 'lat' #'volume_pa'trycpp

        self.func = 'loop'

        self.test_shuffle = 0
        self.method = 'noTarget'
        self.rmseORmaxae = 'rmse'
        
        self.repeatlst = None #[(0,19)]
        
        self.trainrlst = [0] #,0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.25] #[0.066,0.1,0.133,0.166,0.2,0.233,0.266,0.3]
        self.hyperplst = [0] #list(range(2,21)) #list(range(2,16)) #[5,6] #list(range(5,8)) #list(range(5,21)) #list(range(7)) if self.testvar==0 else [0,1] #list(range(2,21)) #,3,4,5,6,7,8,9,10] #,15,20,25,30,35,40,45,50,55,60,65,70] #list(range(17))
        self.howoften = 30 #30 #30 #20 if self.testvar==0 else 2

        ##############################################################################
        self.daa_parameters = {
            'test'                  : self.testvar, # set to 0 to pass regular testlst
            'test_or_train_arche'   : 'train', # from which set shall arche be collected
            'normalize'             : 'lin_norm', # normalize input features and target to mean 0 stddev 1 iot get consistent results
            'version'               : 'luigi',
            'at_loss_factor'        : 1.0,
            'target_loss_factor'    : 1.0,
            'recon_loss_factor'     : 1.0,
            'kl_loss_factor'        : 1.0,
            'anneal'                : 0
            }
        self.daa_parameters.update({
            'n_epochs'    : 10000,            #  if self.daa_parameters['test']==0 else 200,
            'dim_lat_lst' : list(range(2,11)) #  if self.daa_parameters['test']==0 else [2]
        })
        self.daa_parameters.update({
            'modelparams' : {i: [self.daa_parameters[var] if var!="dim_lat_lst" else i
                                 for var in ['normalize','version','n_epochs','dim_lat_lst','at_loss_factor', 
                                             'target_loss_factor','recon_loss_factor','kl_loss_factor','anneal']] 
                            for i in self.daa_parameters['dim_lat_lst'] }
            })
        ##############################################################################
        #region: set derived parameters
        if self.dataset == 'fcc3_relaxed_PBE_magmNon0_nodupl':
            self.memGB = 500 
        else: self.memGB = None
        self.time = "96:00:00"
        
        self.method = self.method if self.func != 'arche_sisso' else 'arche'
        self.nodes = 1 if self.testvar==0 else 1
        self.desc_dim   = 3
        self.rung       = 2 if not (self.dataset=='tutorial' and self.testvar==0) else 3
        self.thresite   = 200 if self.testvar==0 else 3
        self.maxlooplen = self.thresite
        if self.dataset == 'perovskite':          self.units = ['t']+['a']*4+['b','b']+['c']*10
        elif self.dataset == 'cubic_perovskites': self.units = ['t1','t2','t3','t4']+['a']*4+['b','b']+['c']*10
        elif self.dataset == 'cubic_perovskites_more_features': self.units = ['t']+['a']*6+['c']*10+['d']*2+['e']*2+['f']*2+['h']*4+['i']*2+['b']*6 # NO MISTAKE THAT b this time at the end!
        elif self.dataset == 'cubic_perovskites_extended': self.units = ['t']+['a']*6+['b']*3+['c']*15
        elif self.dataset == 'tutorial':          self.units = ['t']+['a']*6+['c']*4
        elif self.dataset == 'fcc2_relaxed_PBE_magmNon0_nodupl': self.units = ['t']+['a']*6+['c']*4+['d']*4
        elif self.dataset == 'fcc3_relaxed_PBE_magmNon0_nodupl': self.units = ['t']+['a']*9+['c']*6+['d']*6
        elif self.dataset[-1]=="3":               self.units = ['t']+['a']*6+['c']*15+['d']*6
        elif self.dataset[-1]=="2":               self.units = ['t']+['a']*4+['c']*10+['d']*4
        elif self.dataset[-1]=="M":               self.units = ['t']+['a']*4+['c']*10+['d']*2
        
        if self.dataset == 'cubic_perovskites':   self.targetlst = [self.target,[tar for tar in ['lat','bul','o_hirsh','o_center'] if tar!=self.target]]
        else:                                     self.targetlst = [0,[]]        
        
        self.now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")        
        self.str1 = 'test' if self.testvar==1 else 'main'
        home = str(Path.home())
        
        self.datapath = "/home/oehlers/Documents/masterthesis/02-data/data" if mj.machinejob().cluster==0 else os.path.join(home,"data")
        self.preparation_path = "/home/oehlers/Documents/masterthesis/04-results/01-preparation" if mj.machinejob().cluster==0 else os.path.join(home,"preparation")
        self.interm_resultspath = "/home/oehlers/Documents/masterthesis/04-results/02-into-cluster" if mj.machinejob().cluster==0 else os.path.join(home,"02-into-cluster")
        self.sissoSubres_path = "/home/oehlers/Documents/masterthesis/04-results/03-after-cluster/SISSOsubres/" if mj.machinejob().cluster==0 else os.path.join(home,"results/SISSOsubres/")
        self.final_resultspath = "/home/oehlers/Documents/masterthesis/04-results/03-after-cluster" if mj.machinejob().cluster==0 else os.path.join(home,"results")
        self.folderpath = os.path.join(self.final_resultspath,self.func)
        
        self.datafile = os.path.join(self.datapath,self.dataset+".dat")
        self.testlstfile = os.path.join(self.datapath,self.dataset)+"_testlst"
        if self.test_shuffle is not None and self.test_shuffle!=False:
            self.testlstfile = self.testlstfile+"_shuffle"+str(self.test_shuffle)
        
        try: 
            os.mkdir(self.final_resultspath)
        except: pass
        try:
            os.mkdir(self.folderpath)
        except: pass
        try:
            os.mkdir(self.sissoSubres_path)
        except: pass
        try:
            os.mkdir(self.preparation_path)
        except: pass
        #endregion


def readClusterers(trainr,hyperp):
    import clustering,json,os

    p = params()
    
    #for new in +[5]:
    new = hyperp
    dfset = clustering.get_dfset_for_clustering_in_primary_space(p)
    clusterer = dfset.get_clusterer(p.method,min_cluster_size=new,save_at=p.preparation_path)
    
    clusterer.get_offsplitMaterialsDicts_forMultiSISSO()
    clusterer.get_merged_clusterMaterialsDict()
    clusterer.save(p.preparation_path)
    # Serialize data into file:
    json.dump( clusterer.offsplitMaterialsDicts_forMultiSISSO,
               open( os.path.join(p.preparation_path,"offsplitDict_{}_{}".format(new,p.method) ), 'w' ) )

def useClusterers(trainr,hyperp):
    import json,os,clustering,myfuncs,mysis

    p = params()
    dfset = clustering.get_dfset_for_clustering_in_primary_space(p)
    
    offsplitDicts = json.load( open(
        os.path.join(p.preparation_path,"offsplitDict_{}_{}".format(hyperp,p.method) )
        ,'r' ) )
    
    print(hyperp)
    print(offsplitDicts.keys())

    for n_offsplits,clusterMaterialsDict in offsplitDicts.items():
        ite = n_offsplits
        dfset = clustering.get_dfset_for_sisso(p.datafile,p.units,p.targetlst,p.testlstfile)
        dfset = dfset.MultiTask(clusterMaterialsDict)
        filename = myfuncs.folder_res_name(locals(),dfset)
        
        sis = mysis.sisso(dfset,p.desc_dim,p.rung,p.rmseORmaxae,folderid=filename,path=p.final_resultspath)
        sis.trysave(filename,p.folderpath) 

def daaPrepare(trainr,hyperp):
    import pandas as pd
    import numpy as np
    from func_collection import linnorm
    from myfuncs import allow_kwargs
    import pickle,os,results_daa,random
    
    p = params()

    datadf = pd.read_csv(p.datafile,sep = " ",index_col=0) 
    
    drop_cols = [col for col in datadf.columns if any(possible_col in col for possible_col in ['bul', 'o_hirsh', 'o_center','Unnamed'])]
    datadf = datadf.drop(drop_cols,1)
    
    print("datadf head")
    print(datadf.head())
    if p.dataset!="cubic_perovskites":
        assert len(datadf.columns)==len(p.units)
    
    normeddf = datadf.apply(linnorm,axis=0)
    allmat = list(normeddf.index)
    
    # here: use other testlsts:
    testlst_name = "{}_testlst_shuffle{}".format(p.dataset,hyperp)
    if testlst_name not in os.listdir(p.datapath):
        with open(p.testlstfile,"rb") as file:
            testlst = pickle.load(file)
        if hyperp > 0:
            random.shuffle(allmat)
            testlst = sorted(allmat[:len(testlst)])
            
            try: os.makedirs(os.path.join(p.preparation_path,"results_collections"))
            except: pass
    
    else:
        with open(os.path.join(p.datapath,testlst_name),'rb') as f:
            testlst = pickle.load(f)
            
    with open(os.path.join(p.preparation_path,"results_collections",testlst_name),'wb') as f:
        pickle.dump(testlst,f)
    
    worklst = sorted([mat for mat in allmat if mat not in testlst])
        
    X_train, X_test = normeddf.drop(p.target,axis=1).loc[worklst,:], normeddf.drop(p.target,axis=1).loc[testlst,:]
    y_train, y_test = normeddf.loc[worklst,p.target], normeddf.loc[testlst,p.target]

    [X_train, X_test, y_train, y_test] = [np.array(df) for df in [X_train, X_test, y_train, y_test]] 
    datadict = {'train_feat':   X_train,
                'train_targets':y_train,
                'test_feat':    X_test,
                'test_targets': y_test
                }
    
    for dim_lat in p.daa_parameters['dim_lat_lst']:
        for ite in range(p.howoften):
            print("LOOK HERE",dim_lat,ite,hyperp)
            allow_kwargs(results_daa.collect_results_master)(
                        datadict,p.preparation_path,hyperp,p.dataset,dim_lat=dim_lat,**p.daa_parameters)
       
def daaIntoSISSO(trainr,hyperp):
    import pandas as pd
    from func_collection import select_arche,linnorm
    import pickle,os,matdf,mysis,myfuncs
    
    # shuffle = hyperp, latdim = ite
    
    p = params()
    
    datadf = pd.read_csv(p.datafile,sep = " ",index_col=0) 
    
    drop_cols = [col for col in datadf.columns if any(possible_col in col for possible_col in ['bul', 'o_hirsh', 'o_center','Unnamed'])]
    datadf = datadf.drop(drop_cols,1)
    
    print("datadf head")
    print(datadf.head())
    if p.dataset!="cubic_perovskites":
        assert len(datadf.columns)==len(p.units)
    
    normeddf = datadf.apply(linnorm,axis=0)
    
    shuffle = hyperp
    
    modelparams = p.daa_parameters['modelparams'][2]
    print(modelparams)
    print(shuffle)
    print(p.daa_parameters['dim_lat_lst'])
    res_filename = "{}_daaPrepare_shuffle{}_{}".format(p.dataset,shuffle,modelparams)
    #if os.path.join(p.preparation_path,'results_collections',res_filename)
    arche_sel, cluster_sel = select_arche(p.daa_parameters['dim_lat_lst'],normeddf,modelparams,
                                                   res_filename,p.method,res_path=p.preparation_path)
    print(cluster_sel.keys())
    testlst_name = "{}_testlst_shuffle{}".format(p.dataset,hyperp)
    with open(os.path.join(p.preparation_path,"results_collections",testlst_name),'rb') as f:
        testlst = pickle.load(f)
    
    testrmses_ntasks = pd.DataFrame(index=[1,2,3])

    dfset = matdf.matdf(p.datafile, units=p.units).Targets(*p.targetlst)
    dfset.test = dfset.targeted.loc[testlst,:]
    dfset.val = dfset.test
    dfset.work = dfset.targeted.loc[[mat for mat in list(dfset.targeted.index) if mat not in testlst],:]
    dfset.train = dfset.work
    
    sis_filename = myfuncs.folder_res_name(locals(),dfset)
    sis = mysis.sisso(dfset,p.desc_dim,p.rung,p.rmseORmaxae,folderid=sis_filename)
    
    testrmses_ntasks.loc[:,1] = sis.pred.outdf['test-rmse']
    
    for nlatdim in p.daa_parameters['dim_lat_lst']:
        ite = nlatdim
        dfset = matdf.matdf(p.datafile, units=p.units).Targets(*p.targetlst)
        dfset.test = dfset.targeted.loc[testlst,:]
        dfset.val = dfset.test
        dfset.work = dfset.targeted.loc[[mat for mat in list(dfset.targeted.index) if mat not in testlst],:]
        dfset.train = dfset.work
        dfset.MultiTask(cluster_sel[nlatdim])
        
        sis_filename = myfuncs.folder_res_name(locals(),dfset)
        sis = mysis.sisso(dfset,p.desc_dim,p.rung,p.rmseORmaxae,folderid=sis_filename)
        
        sis.trysave(sis_filename,os.path.join(p.final_resultspath,"daaIntoSISSO"))
        testrmses_ntasks.loc[:,nlatdim+1] = sis.pred.outdf['test-rmse']
    
    testrmses_ntasks.to_csv(os.path.join(p.final_resultspath,"testrmses_{}".format(res_filename)),sep=",")

def HDBSCANderivedSpaceRandom(trainr, hyperp):
    import myHdbscan, mysis, myfuncs, clustering, os, pickle
    p = params()

    dfset_for_clustering = clustering.get_dfset_for_clustering_in_3d_sisso_space(p)
    clusterer = dfset_for_clustering.get_clusterer("noTarget", min_cluster_size=5)
    offsplitMaterialsDicts_forMultiSISSO = clusterer.get_offsplitMaterialsDicts_forMultiSISSO(rigid_lambda=False)

    for lam, clusterMaterialsDict in offsplitMaterialsDicts_forMultiSISSO.items():
        hyperp = len([clusternr for clusternr in clusterMaterialsDict if clusternr!=-1])
        for ite in range(p.howoften):
            dfset_for_sisso = clustering.get_dfset_for_sisso(p.datafile, p.units, p.targetlst, p.testlstfile)
            dfset_for_sisso = dfset_for_sisso.assign_trainpts_randomly(clusterMaterialsDict,trainr)

            filename = myfuncs.folder_res_name(locals(), dfset_for_sisso) + "_testlstshuffle" + str(p.test_shuffle)

            already_done_dict = pickle.load(open(os.path.join(p.final_resultspath, 'already_done'), 'rb'))
            if p.func not in list(already_done_dict.keys()) \
                    or filename not in pickle.load(open(os.path.join(p.final_resultspath, 'already_done'), 'rb'))[p.func]:

                try:
                    sis = mysis.sisso(dfset_for_sisso, p.desc_dim, p.rung, p.rmseORmaxae, folderid=filename,
                                      path=p.final_resultspath)
                    sis.trysave(filename, p.folderpath)
                except:
                    pass

def kmeansDerivedSpaceRandom(trainr, hyperp):
    import myKmeans, mysis, myfuncs, clustering, os, pickle
    p = params()
    # trainr == trainr
    # hyperp as fed to function has no meaning, will be reassigned to nclust in range(2,21)
    # ite == ite

    if p.dataset == "cubic_perovskites_extended":
        raise Exception("Not programmed yet for extended cubic perovskites!")

    for nclust in range(2,21):
        hyperp = nclust
        for ite in range(p.howoften):
            dfset_for_clustering = clustering.get_dfset_for_clustering_in_3d_sisso_space(p)
            kmeanss = dfset_for_clustering.get_kmeans(hyperp, "noTarget")

            dfset_for_sisso = clustering.get_dfset_for_sisso(p.datafile, p.units, p.targetlst, p.testlstfile)
            dfset_for_sisso = dfset_for_sisso.assign_trainpts_randomly(kmeanss.materialDict,trainr)

            filename = myfuncs.folder_res_name(locals(), dfset_for_sisso) + "_testlstshuffle" + str(p.test_shuffle)

            already_done_dict = pickle.load(open(os.path.join(p.final_resultspath, 'already_done'), 'rb'))
            if p.func not in list(already_done_dict.keys()) \
                    or filename not in pickle.load(open(os.path.join(p.final_resultspath, 'already_done'), 'rb'))[p.func]:
                try:
                    sis = mysis.sisso(dfset_for_sisso, p.desc_dim, p.rung, p.rmseORmaxae, folderid=filename,
                                      path=p.final_resultspath)
                    sis.trysave(filename, p.folderpath)
                except:
                    pass

def kmeansCenter(trainr,hyperp):
    import myfuncs,mysis,clustering,myKmeans

    p = params() 

    dfset = clustering.get_dfset(p.datafile,p.units,p.targetlst,p.testlstfile)
    print(dfset.targeted.head())
    
    for ite in range(p.howoften):
        kmeans = dfset.get_kmeans(hyperp,p.method)
        if hyperp==1: 
            clusterdict = {1: [list(dfset.work.index),list(dfset.test.index)]}
            dfset = dfset.assign_trainpts_randomly(clusterdict,trainr)
        else: 
            dfset = dfset.assign_clusterCenters_as_trainpts(kmeans)
        filename = myfuncs.folder_res_name(locals(),dfset)
        
        sis = mysis.sisso(dfset,p.desc_dim,p.rung,p.rmseORmaxae,folderid=filename,path=p.final_resultspath)
        sis.trysave(filename,p.folderpath)


def kmeansDerivedSpaceCenter(trainr, hyperp):
    import myfuncs, mysis, clustering, myKmeans
    from copy import copy
    p = params()
    # trainr not used, set to [0]!
    # hyperp = nclusters
    dfset_for_clustering = clustering.get_dfset_for_clustering_in_3d_sisso_space(p)

    for ite in range(p.howoften):
        kmeans = copy(dfset_for_clustering).get_kmeans(hyperp, 'noTarget')
        dfset_for_sisso = clustering.get_dfset_for_sisso(p.datafile, p.units, p.targetlst, p.testlstfile
                                                         ).assign_clusterCenters_as_trainpts(kmeans)
        filename = myfuncs.folder_res_name(locals(), dfset_for_sisso)+"_testlstshuffle"+str(p.test_shuffle)
        #print(sorted(dfset_for_sisso.train.index))
        try:
            sis = mysis.sisso(dfset_for_sisso, p.desc_dim, p.rung, p.rmseORmaxae, folderid=filename, path=p.final_resultspath)
            sis.trysave(filename, p.folderpath)
        except:
            pass


def SISSOhdbscanMultiSISSO(trainr, hyperp):
    import myfuncs, mysis, clustering, myHdbscan,matdf,os,pickle
    from copy import copy
    import pandas as pd
    p = params()
    dfset_for_clustering = clustering.get_dfset_for_clustering_in_3d_sisso_space(p,coefs_stretch=True)
    if hyperp is None or hyperp == 0:
        hyperp = None
    clusterer = dfset_for_clustering.get_clusterer(p.method, min_cluster_size=hyperp, save_at=p.preparation_path)
    offsplitMaterialsDicts_forMultiSISSO = clusterer.get_offsplitMaterialsDicts_forMultiSISSO(rigid_lambda=False)

    for n_offsplits, clusterMaterialsDict in offsplitMaterialsDicts_forMultiSISSO.items():
        ite = n_offsplits
        dfset_for_sisso = clustering.get_dfset_for_sisso(p.datafile, p.units, p.targetlst, p.testlstfile)
        dfset_for_sisso = dfset_for_sisso.MultiTask(clusterMaterialsDict)

        filename = myfuncs.folder_res_name(locals(), dfset_for_sisso) + "testlst_shuffle" + str(p.test_shuffle)
        if filename not in pickle.load(open(os.path.join(p.re))):
            try:
                sis = mysis.sisso(dfset_for_sisso, p.desc_dim, p.rung, p.rmseORmaxae, folderid=filename,
                                  path=p.final_resultspath)
                sis.trysave(filename, p.folderpath)
            except:
                pass

def SISSOkmeansMultiSISSO(trainr, hyperp):
    import myfuncs, mysis, clustering, myKmeans,matdf,os
    from copy import copy
    import pandas as pd
    p = params()

    for ite in range(p.howoften):
        filename = myfuncs.folder_res_name(locals())
        if not filename in os.listdir(p.folderpath):
            i=0
            while i<5:
                try:
                    dfset_for_kmeans = clustering.get_dfset_for_clustering_in_3d_sisso_space(p,coefs_stretch=True)

                    kmeans = dfset_for_kmeans.get_kmeans(hyperp, p.method)
                    clusterMaterialsDict = kmeans.get_clusterMaterialsDict()

                    dfset_for_sisso = clustering.get_dfset_for_sisso(p.datafile, p.units, p.targetlst, p.testlstfile)
                    dfset_for_sisso = dfset_for_sisso.MultiTask(clusterMaterialsDict)


                    sis = mysis.sisso(dfset_for_sisso, p.desc_dim, p.rung, p.rmseORmaxae, folderid=filename,
                                      path=p.final_resultspath)
                    sis.trysave(filename, p.folderpath)
                    i = 5
                except:
                    i += 1

def kmeansMultiSISSO(trainr,hyperp):
    import myfuncs,mysis,clustering,myKmeans,os
    from copy import copy
    p = params()

    for ite in range(p.howoften):
        filename = myfuncs.folder_res_name(locals())
        if not filename in os.listdir(p.folderpath):
            i=0
            while i<5:
                try:
                    dfset_for_clustering = clustering.get_dfset_for_clustering_in_primary_space(p)
                    kmeans = dfset_for_clustering.get_kmeans(hyperp,p.method)
                    clusterMaterialsDict = kmeans.get_clusterMaterialsDict()

                    dfset_for_sisso = clustering.get_dfset_for_sisso(p.datafile, p.units, p.targetlst, p.testlstfile)
                    dfset_for_sisso = dfset_for_sisso.MultiTask(clusterMaterialsDict)

                    sis = mysis.sisso(dfset_for_sisso,p.desc_dim,p.rung,p.rmseORmaxae,folderid=filename,path=p.final_resultspath)
                    sis.trysave(filename,p.folderpath)
                    i = 5
                except:
                    i += 1


def randomSubsetHDBSCANDerivedSpace(trainr,hyperp):
    import myfuncs, mysis, clustering, os, myHdbscan,pickle
    from copy import copy

    p = params()
    TRAINR_NOT_USED = trainr

    dfset_for_hdbscan = clustering.get_dfset_for_clustering_in_3d_sisso_space(p)

    hdbscan_clusterer = dfset_for_hdbscan.get_clusterer(p.method)
    nclust2clusterDicts = hdbscan_clusterer.get_nclusters2clusterDicts()

    for nclust, clusterDict in nclust2clusterDicts.items():
        hyperp = copy(nclust)
        set_clusternr = list(set(list(clusterDict.keys())) - {-1})
        max_npicked_per_cluster = min(len(clusterDict[clusternr][0])
                                      for clusternr in set_clusternr)
        for trainr in range(1,1+max_npicked_per_cluster):
            npicked_per_cluster = copy(trainr)

            dfset_for_sisso = clustering.get_dfset_for_sisso(p.datafile, p.units, p.targetlst, p.testlstfile)
            dfset_for_sisso = dfset_for_sisso.pick_randomly_uniform_among_clusters(clusterDict,npicked_per_cluster)
            filename = myfuncs.folder_res_name(locals(), dfset_for_sisso)

            already_done_dict = pickle.load(open(os.path.join(p.final_resultspath, 'already_done'), 'rb'))
            if p.func not in list(already_done_dict.keys()) \
                    or filename not in pickle.load(open(os.path.join(p.final_resultspath,'already_done'),'rb') )[p.func]:
                try:
                    sis = mysis.sisso(dfset_for_sisso, p.desc_dim, p.rung, p.rmseORmaxae,
                                      folderid=filename, path=p.final_resultspath)
                    sis.trysave(filename, p.folderpath)
                except:
                    pass



def randomSubsetHDBSCANPrimarySpace(trainr,hyperp):
    import myfuncs, mysis, clustering, os, myHdbscan,pickle
    from copy import copy

    p = params()
    TRAINR_NOT_USED = trainr

    dfset_for_hdbscan = clustering.get_dfset_for_clustering_in_primary_space(p)

    hdbscan_clusterer = dfset_for_hdbscan.get_clusterer(p.method)
    nclust2clusterDicts = hdbscan_clusterer.get_nclusters2clusterDicts()

    for nclust, clusterDict in nclust2clusterDicts.items():
        hyperp = copy(nclust)
        set_clusternr = list(set(list(clusterDict.keys())) - {-1})
        max_npicked_per_cluster = min(len(clusterDict[clusternr][0])
                                      for clusternr in set_clusternr)
        for trainr in range(1,1+max_npicked_per_cluster):
            npicked_per_cluster = copy(trainr)

            dfset_for_sisso = clustering.get_dfset_for_sisso(p.datafile, p.units, p.targetlst, p.testlstfile)
            dfset_for_sisso = dfset_for_sisso.pick_randomly_uniform_among_clusters(clusterDict,npicked_per_cluster)
            filename = myfuncs.folder_res_name(locals(), dfset_for_sisso)

            already_done_dict = pickle.load(open(os.path.join(p.final_resultspath, 'already_done'), 'rb'))
            if p.func not in list(already_done_dict.keys()) \
                    or filename not in pickle.load(open(os.path.join(p.final_resultspath,'already_done'),'rb') )[p.func]:

                try:
                    sis = mysis.sisso(dfset_for_sisso, p.desc_dim, p.rung, p.rmseORmaxae,
                                      folderid=filename, path=p.final_resultspath)
                    sis.trysave(filename, p.folderpath)
                except:
                    pass


def randomSubsetKmeansPrimarySpace(trainr,hyperp):
    import myfuncs, mysis, clustering, os, myKmeans,pickle
    from copy import copy

    p = params()
    TRAINR_NOT_USED = trainr

    for ite in range(p.howoften):
        dfset_for_kmeans = clustering.get_dfset_for_clustering_in_primary_space(p)
        kmeans = dfset_for_kmeans.get_kmeans(hyperp, p.method)

        set_labels = list(set(list(kmeans.workClusterDf['labels'])) - {-1})
        max_npicked_per_cluster = min(kmeans.workClusterDf[kmeans.workClusterDf['labels']==label].shape[0]
                                      for label in set_labels)
        max_total = 150
        multi = 5 if hyperp == 2 else 3 if hyperp in [3, 4] else 2 if hyperp in [5, 6, 7, 8, 9] else 1

        rrange = [i * multi for i in list(range(1, 2 + int(float(max_npicked_per_cluster) / float(multi))))
                  if i * hyperp * multi < max_total]
        for trainr in rrange:
            npicked_per_cluster = copy(trainr)

            dfset_for_sisso = clustering.get_dfset_for_sisso(p.datafile, p.units, p.targetlst, p.testlstfile)
            dfset_for_sisso = dfset_for_sisso.pick_randomly_uniform_among_clusters(kmeans,npicked_per_cluster)
            filename = myfuncs.folder_res_name(locals(), dfset_for_sisso)

            already_done_dict = pickle.load(open(os.path.join(p.final_resultspath, 'already_done'), 'rb'))
            if p.func not in list(already_done_dict.keys()) \
                    or filename not in pickle.load(open(os.path.join(p.final_resultspath,'already_done'),'rb') )[p.func]:

                try:
                    sis = mysis.sisso(dfset_for_sisso, p.desc_dim, p.rung, p.rmseORmaxae, folderid=filename, path=p.final_resultspath)
                    sis.trysave(filename, p.folderpath)
                except:
                    pass


red = ["Li", "Na", "K", "Rb", "Cs"]
orange = ["Be", "Mg", "Ca", "Sr", "Ba"]
lachsA = ['Sc', "Y"]
pink = ["La", "Ce", "Pr", "Nd", "Pm", "Sm"]
A = red + orange + pink + lachsA
more = ['Al', 'Co', 'Cu', 'Ge', 'Mn', 'Mo', 'Pt', 'Rh', 'Sb', 'V', 'W']
lachsB = ["Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu",
          "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag",
          "Ta", "W", "Pt"]
grey = ["Zn", "Cd", 'Al', "Ga", "Sn", "Pb", "Bi"]
green = ["Ge", "Sb"]
B = lachsB + grey + green

def randomSubsetElB(trainr,hyperp):
    """try educated guess:
    randomly select materials in elBgroup, but in such a way that elA distribution as uniform as possible"""
    import myfuncs, mysis, clustering, myKmeans,re,os,pickle
    from random import shuffle
    from copy import copy
    import pandas as pd

    p = params()

    dfset = clustering.get_dfset_for_sisso(p.datafile, p.units, p.targetlst, p.testlstfile)

    idxs = [[el[1], el[0]] for el in
            [re.findall('[A-Z][^A-Z]*', mat.replace("O3", ""))
             for mat in list(dfset.work.index)]]
    b2A = {b: sorted([mat[1] for mat in idxs if mat[0] == b])
           for b in B}
    b2workmats = {b: sorted([mat[1]+mat[0]+"O3" for mat in idxs if mat[0] == b])
           for b in B}
    b2nworkmats = {b: len([mat for mat in idxs if mat[0] == b])
                   for b in B}
    max_ntrain_per_elB = min(b2nworkmats.values())
    # assert 1 <= trainr <= max_ntrain_per_elB and isinstance(trainr,int), "trainr must be number of materials selected for training per elBgroup"
    testrmses3d_ite_ntrain = pd.DataFrame(index=list(range(p.howoften)),
                                          columns=list(range(5, dfset.work.shape[0] + 1)))
    testrmses3d_ite_ntrain.index.name = 'ite'
    def get_shuffled(lst):
        assert isinstance(lst, (list, tuple))
        copylst = copy(lst)
        shuffle(copylst)
        return copylst

    for trainr in range(1, 1 + max_ntrain_per_elB):
        for ite in range(p.howoften):
            selectedTrainingMaterials = []
            for b in B:
                selectedTrainingMaterials += get_shuffled(b2workmats[b])[:trainr]
            dfset.train = dfset.work.loc[selectedTrainingMaterials, :]
            dfset.val = dfset.work.loc[[mat for mat in dfset.work.index if mat not in selectedTrainingMaterials], :]
            print(len(B),trainr, dfset.train.shape)
            assert dfset.train.shape[0] == trainr * 28, 'traindf shape must be trainr*28'

            filename = myfuncs.folder_res_name(locals(), dfset)
            already_done_dict = pickle.load(open(os.path.join(p.final_resultspath, 'already_done'), 'rb'))
            if p.func not in list(already_done_dict.keys()) \
                    or filename not in pickle.load(open(os.path.join(p.final_resultspath,'already_done'),'rb') )[p.func]:

                try:
                    sis = mysis.sisso(dfset, 3, 2, 'rmse')
                    sis.trysave(filename, p.folderpath)
                    testrmses3d_ite_ntrain.loc[ite, trainr] = sis.pred.outdf.loc[3, 'test-rmse']
                    testrmses3d_ite_ntrain.to_csv(os.path.join(p.folderpath, 'testrmses3d_ite_ntrain'), sep=',')
                except:
                    pass

def randomSubsetGroupB(trainr,hyperp):
    import myfuncs, mysis, clustering, myKmeans,re,os,pickle
    from random import shuffle
    from copy import copy
    import pandas as pd
    import numpy as np
    import itertools as itt

    p = params()
    dfset = clustering.get_dfset_for_sisso(p.datafile, p.units, p.targetlst, p.testlstfile)

    groupsB = [tuple(sorted(blst)) for blst in [['Ti','Zr'],['V',"Nb","Ta"],["Cr","Mo","W"],["Mn","Tc"],["Fe","Ru"],["Co","Rh"],["Ni","Pd","Pt"],
           ["Cu","Ag"],["Zn","Cd"],["Al","Ga"],["Ge","Sn","Pb"],["Sb","Bi"]]]

    idxs = [[el[1], el[0]] for el in
            [re.findall('[A-Z][^A-Z]*', mat.replace("O3", "")) for mat in list(dfset.work.index)]]
    b2workA = {b: sorted([mat[1] for mat in idxs if mat[0] == b])
                   for b in B}
    g2workA = {g: sorted([mat[1] for mat in idxs if mat[0] in g])
                   for g in groupsB}
    g2workmats = {g: sorted([mat[1]+mat[0]+"O3" for mat in idxs if mat[0] in g])
               for g in groupsB}
    b2nworkmats = {b: len([mat for mat in idxs if mat[0] == b])
                     for b in B}
    g2nworkmats = {groupB: np.sum([b2nworkmats[b] for b in groupB])
                  for groupB in groupsB}
    max_ntrain_per_groupB = min(g2nworkmats.values())

    def get_shuffled(lst):
        assert isinstance(lst, (list, tuple))
        copylst = copy(lst)
        shuffle(copylst)
        return copylst

    testrmses3d_ite_ntrain = pd.DataFrame(index=list(range(p.howoften)),
                                          columns=list(range(5, dfset.work.shape[0] + 1)))
    testrmses3d_ite_ntrain.index.name = 'ite'
    for trainr in range(1, 1 + max_ntrain_per_groupB):
        for ite in range(p.howoften):
            selectedTrainingMaterials = []
            for g in groupsB:
                selectedTrainingMaterials += get_shuffled(g2workmats[g])[:trainr]
            dfset.train = dfset.work.loc[selectedTrainingMaterials, :]
            dfset.val = dfset.work.loc[[mat for mat in dfset.work.index
                                        if mat not in selectedTrainingMaterials], :]

            assert dfset.train.shape[0] == trainr * len(groupsB), 'traindf shape must be trainr*28'

            filename = myfuncs.folder_res_name(locals(), dfset)
            already_done_dict = pickle.load(open(os.path.join(p.final_resultspath, 'already_done'), 'rb'))
            if p.func not in list(already_done_dict.keys()) \
                    or filename not in pickle.load(open(os.path.join(p.final_resultspath,'already_done'),'rb') )[p.func]:

                try:
                    sis = mysis.sisso(dfset, 3, 2, 'rmse')
                    sis.trysave(filename, p.folderpath)
                    testrmses3d_ite_ntrain.loc[ite, trainr] = sis.pred.outdf.loc[3, 'test-rmse']
                    testrmses3d_ite_ntrain.to_csv(os.path.join(p.folderpath, 'testrmses3d_ite_ntrain'), sep=',')
                except:
                    pass

def guidedRandomSubsetElA(trainr,hyperp):
    """try educated guess:
    randomly select materials in elAgroup, but in such a way that elB distribution as uniform as possible"""
    import myfuncs, mysis, clustering, myKmeans,re,os
    from random import shuffle
    from copy import copy
    import pandas as pd

    p = params()

    dfset = clustering.get_dfset_for_sisso(p.datafile, p.units, p.targetlst, p.testlstfile)

    idxs = [[els[0], els[1]] for els in
            [re.findall('[A-Z][^A-Z]*', mat.replace("O3", "")) for mat in list(dfset.work.index)]]
    a2B = {a: sorted([mat[1] for mat in idxs if mat[0] == a])
                   for a in A}

    a2nworkmats = {a: len([mat for mat in idxs if mat[0] == a])
                     for a in A}
    max_ntrain_per_elA = min(a2nworkmats.values())
    #assert 1 <= trainr <= max_ntrain_per_elB and isinstance(trainr,int), "trainr must be number of materials selected for training per elBgroup"
    testrmses3d_ite_ntrain = pd.DataFrame(index=list(range(p.howoften)))
    testrmses3d_ite_ntrain.index.name = 'ite'

    def get_shuffled(lst):
        assert isinstance(lst, (list, tuple))
        copylst = copy(lst)
        shuffle(copylst)
        return copylst

    for trainr in range(1,1+max_ntrain_per_elA):
        for ite in range(p.howoften):
            shuffle(A)
            b2npicked = {b: 0 for b in B}
            selectedTrainingMaterials = []
            print(A)
            for a in A:
                Bofa = copy(a2B[a])
                npicked2shuffledb = {
                    this_npicked: get_shuffled([b for b, npicked in b2npicked.items()
                                                if npicked == this_npicked])
                    for this_npicked in set(list(b2npicked.values()))}
                pick = [b for npicked in sorted(npicked2shuffledb.keys())
                        for b in npicked2shuffledb[npicked]
                        if b in Bofa][:trainr]
                for b in pick:
                    b2npicked[b] += 1
                    selectedTrainingMaterials += [a + b + "O3"]
            print(sorted(selectedTrainingMaterials))
            dfset.train = dfset.work.loc[selectedTrainingMaterials,:]
            dfset.val = dfset.work.loc[[mat for mat in dfset.work.index
                                        if mat not in selectedTrainingMaterials],:]

            assert dfset.train.shape[0] == trainr*18, 'traindf shape must be trainr*18'

            filename = myfuncs.folder_res_name(locals(),dfset)+"_testshuffle"+str(p.test_shuffle)
            try:
                sis = mysis.sisso(dfset,3,2,'rmse')
                sis.trysave(filename,p.folderpath)
                testrmses3d_ite_ntrain.loc[ite,trainr] = sis.pred.outdf.loc[3,'test-rmse']
                testrmses3d_ite_ntrain.to_csv(os.path.join(p.folderpath, 'testrmses3d_ite_ntrain'), sep=',')
            except:
                pass

def guidedRandomSubsetGroupB(trainr,hyperp):
    import myfuncs, mysis, clustering, myKmeans,re,os
    from random import shuffle
    from copy import copy
    import pandas as pd
    import numpy as np
    import itertools as itt

    p = params()
    dfset = clustering.get_dfset_for_sisso(p.datafile, p.units, p.targetlst, p.testlstfile)

    groupsB = [tuple(sorted(blst)) for blst in [['Ti','Zr'],['V',"Nb","Ta"],["Cr","Mo","W"],["Mn","Tc"],["Fe","Ru"],["Co","Rh"],["Ni","Pd","Pt"],
           ["Cu","Ag"],["Zn","Cd"],["Al","Ga"],["Ge","Sn","Pb"],["Sb","Bi"]]]

    idxs = [[el[1], el[0]] for el in
            [re.findall('[A-Z][^A-Z]*', mat.replace("O3", "")) for mat in list(dfset.work.index)]]
    b2workA = {b: sorted([mat[1] for mat in idxs if mat[0] == b])
                   for b in B}
    g2workA = {g: sorted([mat[1] for mat in idxs if mat[0] in g])
                   for g in groupsB}
    b2nworkmats = {b: len([mat for mat in idxs if mat[0] == b])
                     for b in B}
    g2nworkmats = {groupB: np.sum([b2nworkmats[b] for b in groupB])
                  for groupB in groupsB}
    max_ntrain_per_groupB = min(g2nworkmats.values())

    def get_shuffled(lst):
        assert isinstance(lst, (list, tuple))
        copylst = copy(lst)
        shuffle(copylst)
        return copylst

    testrmses3d_ite_ntrain = pd.DataFrame()
    for trainr in range(1,1+max_ntrain_per_groupB):
        for ite in range(p.howoften):
            shuffle(groupsB)
            a2npicked = {a: 0 for a in A}
            b2npicked = {b: 0 for b in B}
            selectedTrainingMaterials = []
            for roundno in range(trainr):
                for g in groupsB:
                    npicked2shuffleda = {k: v
                                         for k,v in {this_npicked: get_shuffled([a for a,npicked in a2npicked.items()
                                                                     if npicked==this_npicked and a in g2workA[g]])
                                                      for this_npicked in set(list(a2npicked.values()))}.items()
                                         if v }
                    npicked2shuffledb = {k: v
                                         for k,v in {this_npicked: get_shuffled([b for b,npicked in b2npicked.items()
                                                                     if npicked==this_npicked and b in g])
                                                      for this_npicked in set(list(b2npicked.values()))}.items()
                                         if v }

                    minpickeda = npicked2shuffleda[min(list(npicked2shuffleda.keys()))]
                    minpickedb = npicked2shuffledb[min(list(npicked2shuffledb.keys()))]
                    nextminpickeda = npicked2shuffleda[min(list(npicked2shuffleda.keys())) + 1] if min(
                        list(npicked2shuffleda.keys())) + 1 in npicked2shuffleda.keys() else []

                    nextminpickedb = npicked2shuffledb[min(list(npicked2shuffledb.keys())) + 1] if min(
                        list(npicked2shuffledb.keys())) + 1 in npicked2shuffledb.keys() else []

                    pick = [(a,b) for a,b in itt.product(minpickeda,minpickedb) if a+b+"O3" in list(set(dfset.work.index) - set(selectedTrainingMaterials))] \
                    + get_shuffled([(a,b) for a,b in itt.product(nextminpickeda,minpickedb) if a+b+"O3" in list(set(dfset.work.index) - set(selectedTrainingMaterials))]
                                  +[(a,b) for a,b in itt.product(minpickeda,nextminpickedb) if a+b+"O3" in list(set(dfset.work.index) - set(selectedTrainingMaterials))])
                    (a,b) = pick[0]

                    a2npicked[a] += 1
                    b2npicked[b] += 1
                    selectedTrainingMaterials += [a+b+"O3"]

            dfset.train = dfset.work.loc[selectedTrainingMaterials,:]
            dfset.val = dfset.work.loc[[mat for mat in dfset.work.index if mat not in selectedTrainingMaterials],:]

            assert dfset.train.shape[0] == trainr*12, 'traindf shape must be trainr*12'

            filename = myfuncs.folder_res_name(locals(),dfset)
            try:
                sis = mysis.sisso(dfset,3,2,'rmse')
                sis.trysave(filename,p.folderpath)
                testrmses3d_ite_ntrain.loc[ite,trainr] = sis.pred.outdf.loc[3,'test-rmse']
                testrmses3d_ite_ntrain.to_csv(os.path.join(p.folderpath, 'testrmses3d_ite_ntrain'), sep=',')
            except:
                pass

def guidedRandomSubsetTypeA(trainr,hyperp):
    import myfuncs, mysis, clustering, myKmeans,re,os
    from random import shuffle
    from copy import copy
    import pandas as pd
    import numpy as np
    import itertools as itt

    p = params()
    dfset = clustering.get_dfset_for_sisso(p.datafile, p.units, p.targetlst, p.testlstfile)

    typesA = [tuple(sorted(alst)) for alst in [red,orange,pink,lachsA]]

    idxs = [[el[0], el[1]] for el in
            [re.findall('[A-Z][^A-Z]*', mat.replace("O3", "")) for mat in list(dfset.work.index)]]
    a2workB = {a: sorted([mat[1] for mat in idxs if mat[0] == a])
                   for a in A}
    g2workB = {g: sorted([mat[1] for mat in idxs if mat[0] in g])
                   for g in typesA}
    a2nworkmats = {a: len([mat for mat in idxs if mat[0] == a])
                     for a in A}
    g2nworkmats = {groupA: np.sum([a2nworkmats[a] for a in groupA])
                  for groupA in typesA}
    max_ntrain_per_groupA = min(g2nworkmats.values())

    def get_shuffled(lst):
        assert isinstance(lst, (list, tuple))
        copylst = copy(lst)
        shuffle(copylst)
        return copylst

    testrmses3d_ite_ntrain = pd.DataFrame()
    for trainr in range(1,1+max_ntrain_per_groupA):
        for ite in range(p.howoften):
            shuffle(typesA)
            a2npicked = {a: 0 for a in A}
            b2npicked = {b: 0 for b in B}
            selectedTrainingMaterials = []
            for roundno in range(trainr):
                for g in typesA:
                    npicked2shuffledb = {k: v
                                         for k,v in {this_npicked: get_shuffled([b for b,npicked in b2npicked.items()
                                                                     if npicked==this_npicked and b in g2workB[g]])
                                                      for this_npicked in set(list(b2npicked.values()))}.items()
                                         if v }
                    npicked2shuffleda = {k: v
                                         for k,v in {this_npicked: get_shuffled([a for a,npicked in a2npicked.items()
                                                                     if npicked==this_npicked and a in g])
                                                      for this_npicked in set(list(a2npicked.values()))}.items()
                                         if v }

                    minpickeda = npicked2shuffleda[min(list(npicked2shuffleda.keys()))]
                    minpickedb = npicked2shuffledb[min(list(npicked2shuffledb.keys()))]
                    nextminpickeda = npicked2shuffleda[min(list(npicked2shuffleda.keys())) + 1] if min(
                        list(npicked2shuffleda.keys())) + 1 in npicked2shuffleda.keys() else []

                    nextminpickedb = npicked2shuffledb[min(list(npicked2shuffledb.keys())) + 1] if min(
                        list(npicked2shuffledb.keys())) + 1 in npicked2shuffledb.keys() else []

                    pick = [(a,b) for a,b in itt.product(minpickeda,minpickedb)
                            if a+b+"O3" in list(set(dfset.work.index) - set(selectedTrainingMaterials))] \
                    + get_shuffled([(a,b) for a,b in itt.product(nextminpickeda,minpickedb)
                                    if a+b+"O3" in list(set(dfset.work.index) - set(selectedTrainingMaterials))]
                                  +[(a,b) for a,b in itt.product(minpickeda,nextminpickedb)
                                    if a+b+"O3" in list(set(dfset.work.index) - set(selectedTrainingMaterials))])
                    (a,b) = pick[0]

                    a2npicked[a] += 1
                    b2npicked[b] += 1
                    selectedTrainingMaterials += [a+b+"O3"]

            dfset.train = dfset.work.loc[selectedTrainingMaterials,:]
            dfset.val = dfset.work.loc[[mat for mat in dfset.work.index if mat not in selectedTrainingMaterials],:]

            assert dfset.train.shape[0] == trainr*4, 'traindf shape must be trainr*4'

            filename = myfuncs.folder_res_name(locals(),dfset)+"_testlstshuffle"+str(p.test_shuffle)
            if dfset.train.shape[0]>10:
                try:
                    sis = mysis.sisso(dfset,3,2,'rmse')
                    sis.trysave(filename,p.folderpath)
                    testrmses3d_ite_ntrain.loc[ite,trainr] = sis.pred.outdf.loc[3,'test-rmse']
                    testrmses3d_ite_ntrain.to_csv(os.path.join(p.folderpath, 'testrmses3d_ite_ntrain'), sep=',')
                except:
                    pass

def kmeansRandom(trainr,hyperp):
    import myfuncs,mysis,clustering,myKmeans

    p = params() 

    for ite in range(p.howoften):
        dfset = clustering.get_dfset(p.datafile,p.units,p.targetlst,p.testlstfile)
        #print(dfset.targeted.head())
        kmeans = dfset.get_kmeans(hyperp,p.method)
        dfset = dfset.assign_trainpts_randomly(kmeans,trainr)
        filename = myfuncs.folder_res_name(locals(),dfset)
        
        sis = mysis.sisso(dfset,p.desc_dim,p.rung,p.rmseORmaxae,folderid=filename,path=p.final_resultspath)
        sis.trysave(filename,p.folderpath)         

def daaRandom(trainr,hyperp):
    import clustering,pickle,os,myfuncs,mysis
    p = params()
    
    clusterpath = os.path.join(p.interm_resultspath,"archeclusters_bestite_False_2_3_4_5_6_7_8_9_10")
    
    with open(clusterpath,'rb') as f:
        bestclusters_all = pickle.load(f)
    
    dfset = clustering.get_dfset(p.datafile,p.units,p.targetlst,p.testlstfile)
    
    for ite in range(p.howoften):
        if hyperp==1: 
            clusterdict = {1: [list(dfset.work.index),list(dfset.test.index)]}
            dfset = dfset.assign_trainpts_randomly(clusterdict,trainr)
        else: 
            dfset = dfset.assign_trainpts_randomly(bestclusters_all[hyperp],trainr)
        
        filename = myfuncs.folder_res_name(locals(),dfset)
        
        sis = mysis.sisso(dfset,p.desc_dim,p.rung,p.rmseORmaxae, folderid=filename,path=p.final_resultspath)
        sis.trysave(filename,p.folderpath)

def learningCurve(trainr,hyperp):
    import mysis,myfuncs,os,clustering
    import pandas as pd
    from copy import copy
    p = params()

    dfset = clustering.get_dfset_for_sisso(p.datafile, p.units, p.targetlst, p.testlstfile)
    testrmses3d_ite_ntrain = pd.DataFrame(index=list(range(p.howoften)),columns=list(range(5,dfset.work.shape[0]+1)))
    testrmses3d_ite_ntrain.index.name = 'ite'
    for n_train in range(5,dfset.work.shape[0]+1):
        for ite in range(p.howoften):
            trainr = copy(n_train)

            dfset = clustering.get_dfset_for_sisso(p.datafile, p.units, p.targetlst, p.testlstfile)
            dfset = dfset.Train(n_train,seed=None)

            filename = myfuncs.folder_res_name(locals(),dfset)
            try:
                sis = mysis.sisso(dfset, p.desc_dim, p.rung, p.rmseORmaxae, folderid=filename, path=p.final_resultspath)
                sis.trysave(filename, p.folderpath)

                testrmses3d_ite_ntrain.loc[ite,n_train] = sis.pred.outdf.loc[3,'test-rmse']
                testrmses3d_ite_ntrain.to_csv(os.path.join(p.folderpath,'testrmses3d_ite_ntrain'),sep=',')
            except:
                pass

def greedy_loop(trainr,hyperp):
    import pickle,myfuncs,matdf,mysis,methods

    
    p = params()
    
    dfset = matdf.matdf(p.datafile,units=p.units).Targets(*p.targetlst).Test().GreedyTrain(10,subset=False)
    filename = myfuncs.folder_res_name(locals(),dfset)

    
    proxysis = mysis.sisso(dfset,p.desc_dim,p.rung,p.rmseORmaxae,folderid=filename)
    res = matdf.results(p.rmseORmaxae).proxy(proxysis)
    res = res.add(proxysis)
    sis = proxysis
    while res.limes=='-' and res.ite < p.thresite :
        print('trainr:{},hyperp:{},ite:{}'.format(trainr,hyperp,res.ite))
        meth_success = methods.method(p.method,sis,hyperp).Exe()
        sis = mysis.sisso(dfset,p.desc_dim,p.rung,p.rmseORmaxae,folderid=filename,path=p.final_resultspath)
        res.add(sis,meth_success).ana(p.maxlooplen)
        with open('./'+filename, 'wb') as handle:
            pickle.dump(res, handle) #

def loop(trainr,hyperp):
    import pickle,myfuncs,matdf,mysis,methods

    p = params()
    
    dfset = matdf.matdf(p.datafile,units=p.units).Targets(*p.targetlst).Test().Train(1)
    
    filename = myfuncs.folder_res_name(locals(),dfset)
    proxysis = mysis.sisso(dfset,p.desc_dim,p.rung,p.rmseORmaxae,folderid=filename)
    res = matdf.results(p.rmseORmaxae).proxy(proxysis)

    dfset = dfset.Test().Train(trainr)
    
    
    sis = mysis.sisso(dfset,p.desc_dim,p.rung,p.rmseORmaxae,folderid=filename,path=p.final_resultspath)
    res = res.add(sis)
    while res.limes=='-' and res.ite < p.thresite :
        meth_success = methods.method(p.method,sis,hyperp).Exe()
        sis = mysis.sisso(dfset,p.desc_dim,p.rung,p.rmseORmaxae,folderid=filename,path=p.final_resultspath)
        res.add(sis,meth_success).ana(p.maxlooplen)
    with open('./'+filename, 'wb') as handle:
        pickle.dump(res, handle)


def arche_sisso(trainr,hyperp):
    import pickle,matdf,mysis,methods,myfuncs

    p = params()
    
    dfset = matdf.matdf(p.datafile,units=p.units).Targets(*p.targetlst).Test().Train(trainr)
    filename = myfuncs.folder_res_name(locals(),dfset)
    meth_success = methods.Arche(dfset,hyperp)
    if meth_success==1: sis = mysis.sisso(dfset,p.desc_dim,p.rung,p.rmseORmaxae,folderid=filename,path=p.final_resultspath)
    res = matdf.results(p.rmseORmaxae).add(sis,meth_success)
    with open('./'+filename, 'wb') as handle:
        pickle.dump(res, handle)
        
def random(trainr):
    import pickle,matdf,mysis,myfuncs

    p = params()
    res = matdf.results(p.rmseORmaxae)
    dfset = matdf.matdf(p.datafile,units=p.units).Targets(*p.targetlst).Test().Train(trainr, seed=None)
    filename = myfuncs.folder_res_name(locals(),dfset)

    for ite in range(p.thresite):
        dfset = matdf.matdf(p.datafile,units=p.units).Targets(*p.targetlst).Test().Train(trainr, seed=None)
        sis = mysis.sisso(dfset,p.desc_dim,p.rung,p.rmseORmaxae,folderid=filename,path=p.final_resultspath)
        res.add(sis)
    with open('./'+filename, 'wb') as handle:
        pickle.dump(res, handle)
    
###############################################################################################################################
###############################################################################################################################
        
def master():  
    import machinejob as mj
    import os
    from datetime import datetime
    import inspect 
    
    p = params()
    
    if p.func=='learning_curve': 
        if not os.path.exists("../results/outdfs/"):
            os.makedirs("../results/outdfs/")
    
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")        
    path = os.path.join(p.final_resultspath,
                        '{}_{}_{}_{}_{}_{}/'.format(now,p.dataset,p.method,p.target,p.func,p.str1))
    if not os.path.exists(path):
        os.makedirs(path)
    os.chdir(path)
    
    func = globals()[p.func]
    keys = inspect.getfullargspec(func).args
    lstlst1 = [getattr(p,arg+'lst') for arg in keys]
    lstlst2 = p.repeatlst
    
    lstlst = lstlst1 if lstlst2 is None and lstlst1 != [None,None] \
        else lstlst2 if lstlst1 == [None,None] and lstlst2 is not None \
        else None
    
    method_char = p.method[0] if p.method is not None else"-"
    
    job = mj.machinejob(test=p.testvar).Set(metadata='{}-{}-{}'.format(p.dataset[0],method_char,p.testvar),nodes=p.nodes,
                                            memGB = p.memGB, time = p.time)
    job.WriteFileS(params,func,lstlst,many=True).Execute()  

if __name__=='__main__':
    master()

## WHAT TO DO IN ORDER TO BE ABLE TO EXECUTE LOOP.PY NICELY FROM EXEPY OR BETTER MACHINEJOBPY??
## google python exec __name__ once internet!
