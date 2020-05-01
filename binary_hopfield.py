import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import time
import pickle

class binary_hopfield:
    def __init__(self,cell_pop,K,g,tau):
        '''cell_pop=number of individuals per cell
           K = interaction matrix (stochastic)
           q = number of states
           g = acivation function [0,1]->[0,1]
           tau = memory length'''
        
        self.N = cell_pop
        self.K = csr_matrix(K)
        self.g = g
        self.c = 1.0/tau
        self.L = K.shape[0]
        self.reset()
        
    def reset(self):
        self.t=0
        self.u = np.random.uniform(0,1,size=self.L)
                
    def update(self,stochastic=True):
        ps = self.g(self.u)
        if stochastic:
            self.X = np.random.binomial(self.N,p=ps)
            self.u = (1.0 - self.c)*self.u + self.c*self.K@self.X/self.N 
        else:
            self.u = (1.0 - self.c)*self.u + self.c*self.K@ps          
        self.t += 1
        
    def generate_correls(self):
        ones = np.ones(self.L)
        ps = self.g(self.u)
        p = np.outer(ps,ones)
        C = p*p.T +(1-p)*(1-p).T
        #C[np.diag_indices(self.L)]=1
        return C
       
            
    def generate_sample(self, survey_nodes):
        ps = self.g(self.u)
        return np.array([np.random.choice([-1,1],p=[p,1-p]) for p in ps[survey_nodes]])
    
    def generate_vs(self):
        ps = self.g(self.u)
        return ps
    
    
    
#Make activation function which takes whole system as argument
def make_g_potts(beta):
    def g(fs):
        a = np.exp(beta*fs)
        b = np.exp(beta*(1.0-fs))
        return a/(a+b)
    return g


#Make activation function which takes whole system as argument
def make_g_neutral():
    def g(fs):
        return fs
    return g

#Make cell-cell interaction matrix    
def make_K(alpha,Adj):
    npt = Adj.shape[0]
    K = np.zeros((npt,npt))
    for i in range(npt):
        for j in range(npt):
            if Adj[i,j]>0 and i !=j:
                K[i,j]=alpha
                K[j,i]=K[i,j]
    for i in range(npt):
        K[i,i]= 1.0 - K[i].sum()
    return K



def sample_sequence(args):
    
    model = args['model']
    N = args['cell_pop']   
    K = args['K']
    tau = args['tau']
    beta = args['beta']
    
    stoch = True
    if model == 'Potts':
        g = make_g_potts(beta)
        stoch=False
    elif model == 'Neutral':
        g = make_g_neutral()
    else:
        print('Activation not recognised')
        return

    sys = binary_hopfield(N,K,g,tau)
    
    samp_times = args['sample_times']
    nodes = args['survey_nodes']
    samples = []
    states = []
    t=0
    idx=0
    next_time = samp_times[idx]
    steps = samp_times[-1]
    while t <steps+2:
        sys.update(stochastic=stoch)    
        if t==next_time:
            samp = sys.generate_sample(nodes)
            samples.append(samp)
            
            vs = sys.generate_vs()
            states.append(vs)
            
            if idx<len(samp_times)-1:
                idx +=1
                next_time = samp_times[idx]

        t+=1
    return np.array(samples), np.array(states)


def plot_sample(sample,XUK,survey_nodes,XB,dims=(5,6),sz=50,show=True):
    plt.figure(figsize=dims)
    plt.axis('off')
    plt.plot(XB[:,0],XB[:,1],'k',rasterized=True)
    plt.scatter(XUK[survey_nodes,0],XUK[survey_nodes,1], marker='o',c=sample,s=sz,alpha=0.75,cmap='rainbow',vmin=0,vmax=1,rasterized=True)
    if show:
        plt.show()
    
    
