#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 13:39:38 2020

@author: oehlers
"""
import subprocess  #hyperparameters ##TODO write file and import
import pandas as pd 
import myfuncs
import inspect
import time as timee
import itertools as itt
#from loop import testvar

   
class machinejob():
    def __init__(self,test=0):
        self.hostname = subprocess.check_output('hostname')[:5].decode("utf-8") 
        self.cluster=0 if self.hostname=='theob' else 1
        self.test = test
        self.files = []

        if self.cluster==1: 
            self.Default()
            self.Sbatchstr()
        
    def ShowConstraints(self,all=0):
        columns=['partition','nodes','ntasks','time']

        constr = {'talos': pd.DataFrame(data=[['p.talos',80,40,'96:00:00'],\
                                         ['s.talos',1,40,'96:00:00']], columns=columns).set_index(['partition']),
                  'cobra': pd.DataFrame(data=[['tiny',0.5,20,'24:00:00'],\
                                         ['express',32,40,'00:30:00'],\
                                         ['medium',32,40,'24:00:00']], columns=columns).set_index(['partition']),
                  'draco': pd.DataFrame(data=[['small',0.5,16,'24:00:00'],\
                                         ['express',32,32,'00:30:00'],\
                                         ['short',32,32,'04:00:00'],\
                                         ['general',32,32,'24:00:00'],\
                                         ['fat',4,32,'24:00:00']], columns=columns).set_index(['partition'])}

        names = constr.keys() if all==1 or self.hostname=='theob' else [self.hostname]
        for name in names: print('Constraints of {0}\n'.format(name), constr[name])
        return self

    def ShowSettings(self):
        if self.cluster==1:
            print({'jobname':self.jobname,'mail':self.mail,'partition':self.partition,'nodes':self.nodes,'ntasks':self.ntasks,'time':self.time})
            return self
    
    def Default(self):
        if self.cluster==1: 
            settings = pd.DataFrame(data=[[['s.talos','1','20','24:00:00'],['p.talos','1','40','24:00:00']],\
                                          [['express','1','40','00:30:00'],  ['medium','1','40','24:00:00']],\
                                          [['express','1','40','00:30:00'],  ['general','1','32','24:00:00']]],\
                                    columns=['test','main'], index=['ta','co','dr'])
            
            mainORtest = 'main' if self.test==0 else 'test'
            self.partition, self.nodes, self.ntasks, self.time = settings.loc[self.hostname[:2],mainORtest]            
            self.jobname = self.hostname+'hyperpar' ## TODO add here string consisting of hyperpar
            self.mail = "FAIL" ## TODO check which mail types and set most convenient one
            self.memGB = None
        else:
            self.partition = self.nodes = self.ntasks = self.time = self.jobname = self.mail = self.memGB = None
        return self
    
                
    def Set(self,metadata=None,jobname=None,mail=None,partition=None,nodes=None,ntasks=None,time=None,memGB=None):
        if self.cluster==1:
            if metadata  is not None: self.metadata  = self.jobname = metadata
            if jobname   is not None: self.jobname   = jobname
            if mail      is not None: self.mail      = mail
            if partition is not None: self.partition = partition
            if nodes     is not None: self.nodes     = nodes
            if ntasks    is not None: self.ntasks    = ntasks
            if time      is not None: self.time      = time
            if memGB     is not None: self.memGB     = memGB
            if self.hostname=="draco": 
                if self.time!="00:30:00": self.time="24:00:00"
                if self.memGB is not None and self.memGB>120:
                    if self.test==0: 
                        self.partition="fat"
                        self.memGB = 500
                    else: self.memGB = 120 
            if self.hostname=="cobra":
                if self.time!= "00:30:00": self.time="24:00:00"
                if self.memGB is not None and self.memGB>180:
                    if self.test==0: 
                        self.partition="fat"
                        self.memGB = 748
                    else: self.memGB = 180
            self.Sbatchstr()
            
        return self

         
    def WriteFileS(self,shared_params,func,lstlst,many):
        from dill.source import getsource

        """Prints definitions of shared_params and func, along with execution of func for given func_kwargs to file(s).
        DEMANDS: - Inside func, shared_params must be called explicitly, e.g. < p = shared_params >
                 - shared_params and func must be processable by dill.source.getresource
                 - if many==0: arglst is lst of values of func-input-parameters, e.g. for func(a,b,c): arglst=[1,2,3]
                 - if many==1: arglst is lst of values-to-permute-lst of func-input-parameters, e.g. for func(a,b,c): arglst=[['a1','a2'],['b1','b2','b3'],[c1]]
                               -> then, function will be executed for itt.product(arglst)"""
        
        self.func = func
        self.lstlst = lstlst
        
        if self.cluster==1:
            def writefileSwrapper(func,lstlst):
                keys = inspect.getfullargspec(func).args
                filename = self.metadata+'-{}'*(len(keys)+1)
                newfile = filename.format(func.__name__,*lstlst)+".py"
                self.files += [newfile]
                with open(newfile,'a') as file:
                        file.write(getsource(shared_params))
                        file.write(getsource(func))
                        for i in range(len(keys)):
                            file.write('\n{} = {}\n'.format(keys[i],lstlst[i]))
                        file.write(getsource(func).replace("def ","").split(":")[0])
                with open('./print2.txt','a') as ff:
                    ff.write('func'+str(func)+'\nkeys'+str(keys)+'\nlstlst'+str(lstlst))
                    
            if many==False: writefileSwrapper(func,lstlst)
            if many==True:  myfuncs.varyFor(writefileSwrapper,func,lstlst)
        
        return self

    def Execute(self,exefiles=[]):
        if exefiles==[]: exefiles  = self.files
        
        if self.cluster==1:
            for exefile in exefiles:
                self.Set(jobname=exefile)
                with myfuncs.newfile('run.cluster') as batchfile: #
                    batchfile.write(self.sbatchstr)
                    batchfile.write('python3 {}'.format(exefile))
                subprocess.Popen('sbatch run.cluster',shell=True).communicate()
                timee.sleep(1)
        else:
            #exec(open(exefile).read()) 
            for trainr,hyperp in itt.product(*self.lstlst):
                self.func(trainr,hyperp)

    
    def Sbatchstr(self):
        if self.cluster==1:
            if hasattr(self,"memGB") and self.memGB != None and self.hostname!="talos": memstr = "\n#SBATCH --mem={}\n".format(self.memGB*1000)  
            else: memstr = ""
            self.sbatchstr = """#!/bin/bash -l
#SBATCH -o ./djob.out.{0}
#SBATCH -e ./djob.err.{0}
#SBATCH -D ./
#SBATCH -J {0}
#SBATCH --partition={1}
#SBATCH --nodes={2}
#SBATCH --ntasks-per-node={3}{6}
#SBATCH --mail-type={4}
#SBATCH --mail-user=oehlers@fhi-berlin.mpg.de
#SBATCH --time={5}

module load intel impi anaconda
""".format(self.jobname,self.partition,self.nodes,self.ntasks,self.mail,self.time,memstr) if self.cluster==1 else None
        return self


        
