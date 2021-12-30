import subprocess, os, shutil
import pandas as pd
from datetime import datetime
import numpy as np
import machinejob as mj
from random import randint
import pickle,json,myfuncs
from copy import copy
MLindict = {}
MLindict['sisso'] = {'ptype':'target property type 1: continuous for regression,2:categorical for classification',
'ntask':'(target prop:) number of tasks (properties or maps) 1: single-task learning, >1: multi-task learning',
'nsample':'(targt prop:) number of samples for each task (seperate the numbers by comma for ntask >1)',
'task_weighting':'(target prop:) 1: no weighting (tasks treated equally) 2: weighted by #sample_task_i/total_sample.a',
'desc_dim': 'dimension of the descriptor (<=3 for classification)',
'restart': 'set .true. to continue a job that was stopped but not yet finished',
'nsf': 'number of scalar features (one feature is one number for each material)',
'rung':'rung (<=3) of the feature space to be constructed (times of applying the opset recursively)',
'opset':'ONE operator set for feature transformation',
'maxcomplexity':'max feature complexity (number of operators in a feature)',
'dimclass': 'group features according to their dimension/unit; those not in any () are dimensionless',
'maxfval_lb':'features having the max. abs. data value <maxfval_lb will not be selected',
'maxfval_ub':'features having the max. abs. data value >maxfval_ub will not be selected',
'subs_sis':'size of the SIS-selected (single) subspace for each descriptor dimension',
'method':"sparsification operator: 'L1L0' or 'L0'; L0 is recommended!",
'L1L0_size4L0':"If method='L1L0', specify the number of features to be screened by L1 for L0",
'fit_intercept':'fit to a nonzero intercept (.true.) or force the intercept to zero (.false.)',
'metric':'for regression only, the metric for model selection: RMSE,MaxAE',
'nm_output':'number of the best models to output'}

class sisso():
    def __init__(self, matdf, desc_dim, rung, rmseORmaxae, n_nodes=None, folderid=None, path='./', ptype=1, task_weighting=2,
                 restart='.false.', opset="'(+)(-)(*)(/)(exp)(log)(^-1)(^2)(^3)(sqrt)(cbrt)(|-|)'", maxcomplexity=10, maxfval_lb='1e-3',
                 maxfval_ub='1e5', subs_sis=100, method="'L0'", L1L0_size4L0=1, fit_intercept=".true.", metric="'RMSE'", nm_output=100):
        self.ntask = ntask = matdf.n_clusters
        assert "ntask" in locals().keys(),locals().keys()
        indict = {lo[0]:lo[1] for lo in locals().items() if lo[0] not in ['self','rmseORmaxae','folderid','n_nodes','path']} 
        path = os.path.join(path,"SISSOsubres")
        self.fit = fsisso(**indict, n_nodes=n_nodes, folderid=folderid, path=path)
        self.pred = psisso(self.fit, folderid=folderid, path=path, rmseORmaxae=rmseORmaxae)
        #self.metadata = [el.replace("fsisso","fit") if "fsisso" in el else "pred."+el for el in self.pred.metadata]
    
    def trysave(self,name,path='./',outdfORallORjson="outdf"):
        try: os.makedirs(path)
        except: pass

        try: 
            if outdfORallORjson=="outdf": self.pred.outdf.to_csv(os.path.join(path,name),sep=",")
            if outdfORallORjson=="all":   pickle.dump(self,open(os.path.join(path,name),'wb'))
            if outdfORallORjson=="json":  
                with open(os.path.join(path,name),'a') as f :
                    f.write(myfuncs.json_serialize(self))
        except: open(os.path.join(path,name+"_FAILED"),'a').write("failed")
            
class fsisso():
    def __init__(self, matdf, desc_dim, rung, n_nodes=None, folderid=None, path='./', ptype=1, ntask=1, task_weighting=2,
                 restart='.false.', opset="'(+)(-)(*)(/)(exp)(log)(^-1)(^2)(^3)(sqrt)(cbrt)(|-|)'", maxcomplexity=10, maxfval_lb='1e-3',
                 maxfval_ub='1e5', subs_sis=100, method="'L0'", L1L0_size4L0=1, fit_intercept=".true.", metric="'RMSE'",
                 nm_output=100):
        print("Fsisso starting...")
        self.indict = {lo[0]:lo[1] for lo in locals().items() if lo[0] not in ['self','rmseORmaxae','folderid','matdf','n_nodes','path']}
        self.matdf = matdf
        self.folderid = datetime.now().strftime("%d-%m-%Y_%H-%M-%S") if folderid==None else folderid+datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        self.path = path
        self.n_nodes = n_nodes
        self.ntask = ntask
         
        self.Prepare()
        self.ExeSisso()    
        self.CatchOutput()
        self.CleanUp()
        
        #self.metadata = ["matdf","indict","outdf"]

    def Prepare(self):
        self.temp = os.path.join(self.path,"fsisso_{0}".format(self.folderid))
        try: os.makedirs(self.temp)
        except: pass
        print("sisso.fit.temp folder:",os.path.abspath(self.temp))    
    
        self.matdf.train.to_csv('{0}/train.dat'.format(self.temp), sep=" ")
        
        units = list(self.matdf.featuredf['unit'][self.matdf.featuredf['usage']=='feature'])
        unique = [units[i] for i in range(len(units)) if units[i] not in units[:i]]
        self.indict.update({'dimclass': ''.join('({}:{})'.format(units.index(unit)+1, len(units)-units[::-1].index(unit)) for unit in unique),
                            'nsample': self.matdf.get_nsample(),
                            'nsf': self.matdf.train.shape[1] - 1})
            
        with open('{0}/SISSO.in'.format(self.temp),'a') as fsissoin:
            for key in self.indict.keys():
                fsissoin.write(key + '=' + str(self.indict[key]) + "\n")

    def ExeSisso(self):
        exefsisso = "~/bin/SISSO" if mj.machinejob().cluster == 0 else "mpirun -np {} ~/bin/SISSO".format(self.n_nodes) if isinstance(self.n_nodes,int) else "srun ~/bin/SISSO"
        subprocess.run(exefsisso, cwd=self.temp, shell=True, stdout=subprocess.PIPE,check=True)

    def CatchOutput(self):
        self.Outdict()
        self.Modeldict()
        self.Outdf()

    def Outdict(self):
        outdict = {}

        with open("{0}/SISSO.out".format(self.temp), "r") as fsissoout:
            outdict['original-SISSO.out'] = outdict['used-SISSO.out'] = fsissoout.readlines()

        for folder in ['desc_dat', 'feature_space', 'models', 'residual']:
            for file in os.listdir('{0}/{1}'.format(self.temp, folder)):
                with open('{0}/{1}/{2}'.format(self.temp, folder, file), 'r') as openfile:
                    outdict['{0}/{1}'.format(folder, file)] = openfile.readlines()
        self.outdict = outdict

    def Modeldict(self):
        modeldict = {dim: pd.DataFrame(index=list(range(self.indict['nm_output'])),
                                       columns=['train-rmse', 'train-maxae', 'descids', 'descs', 'coefs', 'interc']) \
                     for dim in range(1, self.indict["desc_dim"] + 1)}

        for dim in range(1,1+self.indict['desc_dim']):
            filestr = '{0}/models/top{1}_00{2}d'.format(self.temp, str(self.indict['nm_output']).zfill(4), dim)
    
            with open(filestr,"r") as filemodels:
                for num, line in enumerate(list(filemodels)[1:]):
                    linelst = line.replace("(", "").replace(")", "").replace("************","  **********").split()
                    modeldict[dim].loc[num, 'train-rmse':'descids'] = linelst[1:3] + [tuple(linelst[3:])]
    
            with open('{0}/feature_space/Uspace.name'.format(self.temp), 'r') as fileuspace:
                descdict = {str(num+1): line.partition('  corr=')[0] for num, line in enumerate(fileuspace)}
                for no in modeldict[dim].index:
                    modeldict[dim].loc[no, 'descs'] = tuple(descdict[modeldict[dim].loc[no, 'descids'][d]] for d in range(dim))
    
            with open('{0}_coeff'.format(filestr), "r") as filecoefs:
                for num, line in enumerate(list(filecoefs)[1:]):
                    modeldict[dim].loc[num, 'interc'] = tuple( line.split()[1+ (1+dim)*tasknr] for tasknr in range(self.ntask) )
                    modeldict[dim].loc[num, 'coefs'] = tuple( tuple(line.split()[2+ (1+dim)*tasknr:2+dim+ (1+dim)*tasknr]) 
                                                                    for tasknr in range(self.ntask) )

        self.modeldict = modeldict

    def Outdf(self):
        outdf = pd.DataFrame(index=list(range(1, self.indict["desc_dim"] + 1)),
                             columns=['rank', 'train-rmse', 'train-maxae', 'descids', 'descs', 'coefs', 'interc'])
        for dim in range(1, self.indict["desc_dim"] + 1):
            outdf.loc[dim,:] = [0] + list(self.modeldict[dim].iloc[0,:]) 
        for i in list(outdf.index):
            outdf.loc[i,'columns'] = str(list(self.matdf.train.columns))
            outdf.loc[i,'trainmat'] = str(list(self.matdf.train.index))
            outdf.loc[i,'valmat'] = str(list(self.matdf.val.index))
            outdf.loc[i,'testmat'] = str(list(self.matdf.test.index))
        self.outdf = outdf
        
    def CleanUp(self):
        shutil.rmtree(self.temp)

class psisso():
    def __init__(self,fsisso,rmseORmaxae,folderid=None, path='./'):
        print("Psisso starting...")
        self.fsisso = fsisso
        self.rmseORmaxae = rmseORmaxae
        self.folderid = datetime.now().strftime("%d-%m-%Y_%H-%M-%S") if folderid==None else folderid
        self.path = path
        self.ntask = self.fsisso.ntask
        
        self.success = 0    
        
        while self.success==0: #rank restriction already accounted for in NextModel
            successlst = []
            for tasknr in range(1,self.fsisso.ntask+1):
                self.tasknr = tasknr
                try:
                    self.ForEachTask()
                    successlst += [True] 
                except:
                    successlst += [False]
                    
            self.success = all(successlst)
            
            if self.success==0:
                self.NextModel()
                self.NewSissoOut()
                self.CleanUp

        if self.success == 1:
            self.ChooseModel(rmseORmaxae)
            #self.metadata = ["fsisso.matdf","fsisso.indict","outdf"] #,"valspace.loc[:,'error']","testspace.loc[:,'error']"]
        if self.success == 0:
            pass
            #self.metadata = ["fsisso.matdf","fsisso.indict","fsisso.outdf"]
        try:
            self.outdf2 = self.outdf.T
            self.outdf2 = self.outdf2.sort_index()
        except:
            pass


    def ForEachTask(self):
        for trainORvalORtest in ['train','val','test']:
            setattr(self,trainORvalORtest+'df',getattr(self.fsisso.matdf,trainORvalORtest))
            if getattr(self,trainORvalORtest+'df') is not None:
                
                self.Prepare(trainORvalORtest)
                self.ExeSisso()
                self.CatchOutput(trainORvalORtest)
                self.CleanUp()
    
    def Prepare(self,trainORvalORtest):
        
        if self.fsisso.ntask==1:
            needed_lines = self.fsisso.outdict['used-SISSO.out']
            self.asPredictDat = getattr(self,trainORvalORtest+'df')
        
        if self.fsisso.ntask>1:
            fsissoout_lines = self.fsisso.outdict['used-SISSO.out']
            keywords = ["coefficients_","Intercept_","RMSE,MaxAE_"]
            def keywordcheck(line, keywords,tasknr): 
                if not any(key in line for key in keywords): return True
                else: return any(line.split(key)[-1][:3]==str(tasknr).zfill(3) for key in keywords)
            needed_lines = [line.replace("_"+str(self.tasknr).zfill(3),"_001") for line in fsissoout_lines 
                            if keywordcheck(line,keywords,self.tasknr)]
            
            self.asPredictDat = copy(getattr(self,trainORvalORtest+'df'))
            exten = "task{}".format(self.tasknr)
            self.asPredictDat = self.asPredictDat.loc[[ind for ind in list(self.asPredictDat.index) if exten==ind.split("_")[1]],:]
            self.asPredictDat.index = [ind.replace("_"+exten,"") for ind in list(self.asPredictDat.index)]
        
        self.temp = os.path.join(self.path,"psisso_{0}".format(self.folderid))
        try: os.makedirs(self.temp)
        except: pass
        with open('{0}/SISSO.out'.format(self.temp),'a') as fsissoout:
            fsissoout.write(''.join(needed_lines) )
        self.asPredictDat.index.name = 'materials'
        self.asPredictDat.to_csv('{0}/predict.dat'.format(self.temp), sep=" ")
        with open('{0}/SISSO_predict_para'.format(self.temp),'w') as psissoin:
            psissoin.write(      str(self.asPredictDat.shape[0])\
                           +"\n"+str(self.asPredictDat.shape[1]-1)\
                           +"\n"+str(self.fsisso.indict['desc_dim'])
                           +"\n1")

    def ExeSisso(self):
        exepsisso = "~/bin/SISSO_predict"
        subprocess.check_call(exepsisso, cwd=self.temp, shell=True, stdout=subprocess.PIPE)
        
    def CatchOutput(self,trainORvalORtest):
        outdf = self.fsisso.outdf
        sisspace = {dim: pd.DataFrame(index=self.asPredictDat.index,
                                     columns=['y','pred','error']+['desc'+str(d) for d in range(1,dim+1)])
                   for dim in range(1,1+self.fsisso.indict['desc_dim'])}
        for XY in ['X','Y']:
            with open('{0}/predict_{1}.out'.format(self.temp, XY), 'r') as pred:
                for num, line in enumerate(pred):
                    for dim in range(1,1+self.fsisso.indict['desc_dim']):
                        lastline = 0 if XY=='X' else 1
                        len = self.asPredictDat.shape[0]+1+lastline
                        if num > (dim-1)*len and num < dim*len-lastline:
                            if XY=='X': sisspace[dim].iloc[num-len*(dim-1)-1,3:3+dim] = line.split() 
                            if XY=='Y': sisspace[dim].iloc[num-len*(dim-1)-1,0:3] = line.split()
                        if num == dim*len-lastline and XY=='Y':
                            linepart = line.partition(":")[2].split()
                            #raise Exception("Modfify this, its false!")
                            outdf.loc[dim, trainORvalORtest+'-nmats-task{}'.format(self.tasknr)] = self.asPredictDat.shape[0]
                            outdf.loc[dim, trainORvalORtest+'-rmse-task{}'.format(self.tasknr)] = float(linepart[0])
                            outdf.loc[dim, trainORvalORtest+'-maxae-task{}'.format(self.tasknr)] = float(linepart[1])
                            if self.tasknr==self.ntask:
                                if trainORvalORtest!='train':
                                    outdf.loc[dim, trainORvalORtest+'-rmse'] = ( np.sum([outdf.loc[dim, trainORvalORtest+'-rmse-task{}'.format(task)]**2
                                                                                  *outdf.loc[dim, trainORvalORtest+'-nmats-task{}'.format(task)]
                                                                                for task in range(1,1+self.ntask)])
                                                                         / np.sum([outdf.loc[dim, trainORvalORtest+'-nmats-task{}'.format(task)]
                                                                                   for task in range(1,1+self.ntask)]) )**0.5
                                    outdf.loc[dim, trainORvalORtest+'-maxae'] = max([outdf.loc[dim, trainORvalORtest+'-maxae-task{}'.format(task)]
                                                                                for task in range(1,1+self.ntask)])
        for dim in range(1,1+self.fsisso.indict['desc_dim']):
            if trainORvalORtest!='train':
                assert not sisspace[dim].isnull().values.any()

        if trainORvalORtest=='train':
            pass
            #if self.tasknr==1: self.trainspace = sisspace
            #if self.tasknr>1: self.trainspace = {dim: self.trainspace[dim].append(sisspace[dim]) for dim in sorted(list(sisspace.keys()))}
        if trainORvalORtest=='val':
            if self.tasknr==1: self.valspace = sisspace
            if self.tasknr>1: self.valspace = {dim: self.valspace[dim].append(sisspace[dim]) for dim in sorted(list(sisspace.keys()))}
        if trainORvalORtest=='test':
            if self.tasknr==1: self.testspace = sisspace
            if self.tasknr>1: self.testspace = {dim: self.testspace[dim].append(sisspace[dim]) for dim in sorted(list(sisspace.keys()))}

        self.outdf = outdf
        
    def CleanUp(self):
        shutil.rmtree(self.temp)

    def NextModel(self):
        print('Next model chosen')
        with open('{0}/predict_Y.out'.format(self.temp), 'r') as predy:
            faileddim = predy.read().count('dimension') + 1 if predy.read().count('dimension')!=self.fsisso.indict['desc_dim'] else 1
        newrank = self.fsisso.outdf.loc[faileddim, 'rank']+1
        if newrank < self.fsisso.indict['nm_output']:
            dropcol = [col for col in self.fsisso.outdf.columns if 'val' in col or 'test' in col]
            self.fsisso.outdf = self.fsisso.outdf.drop(dropcol, axis=1)
            self.fsisso.outdf.loc[faileddim, ['rank']+list(self.fsisso.modeldict[faileddim].columns)] = [newrank] + list(self.fsisso.modeldict[faileddim].loc[newrank,:])
        else: 
            self.fsisso.Outdf() # sets outdf back to models of rank 0
            self.success = -1   # stops psisso loop over models, as all possible models tested
            

    def NewSissoOut(self):
        oldfsissoout = self.fsisso.outdict["used-SISSO.out"]
        self.fsisso.outdict["used-SISSO.out"] = []
        enum = enumerate(oldfsissoout)
        outdf = self.fsisso.outdf
        dim_linenum = {}
        for num, line in enum:
            for dim in range(1,1+self.fsisso.indict['desc_dim']):
                if "{0}D descriptor (model):".format(dim) in line: 
                    dim_linenum[dim] = num 
        for num, line in enum:
            newoldlst = []
            for dim in range(1,1+self.fsisso.indict['desc_dim']):
                if num > dim_linenum[dim] and num <= dim_linenum[dim]+3+dim+3*self.ntask:
                    for d in range(dim):
                        if num == dim_linenum[dim]+3+d: newoldlst += [(outdf.loc[dim, 'descids'][d],line.replace(" ","").split(":")[0]),
                                                                      (outdf.loc[dim, 'descs'][d],  line.replace(" ","").split(":")[1])] 
                    self.tasknr = 1
                    while self.tasknr <= self.ntask:
                        exten = str(self.tasknr).zfill(3)
                        pos = self.tasknr-1                        
                        if "coefficients_{}".format(exten) in line: newoldlst += [(outdf.loc[dim, 'coefs'][pos][d],line.split()[d+1]) for d in range(dim)]           
                        if "Intercept_{}".format(exten)    in line: newoldlst += [(outdf.loc[dim, 'interc'][pos],  line.split()[1])]              
                        if "RMSE,MaxAE_{}".format(exten)   in line: newoldlst += [(outdf.loc[dim, 'train-rmse'],   line.split()[1]),
                                                                                  (outdf.loc[dim, 'train-maxae'],  line.split()[2])]    
                    self.tasknr += 1
            newline = line
            for newold in newoldlst:
                newline = newline.replace(newold[1], newold[0])
            self.fsisso.outdict["used-SISSO.out"] += [newline]
    
    def ChooseModel(self,rmseORmaxae):
        bestdim = int(np.array([float(i) for i in self.outdf.loc[:,'val-'+rmseORmaxae]]).argmin())+1
        self.outdf['chosen']=[1 if dim==bestdim else 0 for dim in self.outdf.index]
        self.tomethods = self.valspace[bestdim]

#if __name__ == '__main__': koennte man noch machen
