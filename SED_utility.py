import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

import binary_hopfield as sim

#**********************************
# LOADING SAVING
#**********************************

def save_matrix(folder,stem,param,matrix):
    tu = (stem,param)
    file = '{}_param_{:.3f}.csv'.format(*tu)
    df = pd.DataFrame(matrix)
    df.to_csv(folder/file,header=False,index=False)    
    print(file, ' saved. ')
    
    
def load_matrix(folder,stem,param):
    tu = (stem,param)
    file = '{}_param_{:.3f}.csv'.format(*tu)
    matrix = np.array(pd.read_csv(folder/file,header=None).values) 
    print(file, ' loaded. ')
    return matrix

def load_maps(folder):
    df_Eng = pd.read_csv(folder/"England_Boundary_OS.csv")
    XB = np.array([df_Eng['X'],df_Eng['Y']]).T
    df_cells = pd.read_csv(folder/'England_cells.csv')
    survey_nodes = df_cells[df_cells['Survey']==1]['node'].values
    XUK = np.array([df_cells['X'],df_cells['Y']]).T
    return XB,XUK,survey_nodes

#**********************************
# SED FILE HANDLING
#**********************************

def get_SED_df(folder,file):
    df = pd.read_csv(folder/file)
    df['dep'] = pd.Series(df['dep'],dtype='category')
    df['states'] = 2*df['dep'].cat.codes-1
    
    #Sort by x coordinate
    df.sort_values('x',inplace=True)
    X0 = df['x'].values
    Y0 = df['y'].values
    c0 = df['states'].values
    L = len(df['x'].values)
    #Remove islands
    remove = [6,8,193]
    X=[]
    Y=[]
    states = []
    for i in range(L):
        if i not in remove:
            X.append(X0[i])
            Y.append(Y0[i])
            states.append(c0[i])
    df_new=pd.DataFrame()
    df_new['x']=np.array(X)
    df_new['y']=np.array(Y)
    df_new['states']=np.array(states)
    return df_new

def plot_SED_df(df,XB):
    plt.figure(figsize=(3,4))
    plt.plot(XB[:,0],XB[:,1],'k')
    plt.scatter(df['x'],df['y'],c=df['states'],cmap='rainbow',s=40,alpha=0.5)
    plt.axis('off')
    plt.show()    

def plot_SED_file(folder,file,XB):
    df = get_SED_df(folder,file)
    plot_SED_df(df,XB)


#**********************************
# SIMULATION UTILITIES 
#**********************************

#Convert interaction range to cell interaction strength
def alpha_equiv(sigma,a):
    '''Interaction strength equivalent to given interaction range
    sigma = interaction range
    a = lattice spacing'''
    return 0.5*(sigma/a)**2

#Convert cell interaction strength to interaction range 
def sigma_equiv(alpha,a):
    '''Interaction range equivalent to given interaction strength
    alpha = interaction strength
    a = lattice spacing'''
    return a*np.sqrt(2*alpha)

#Effective interaction range adjusted for smaller population
def pop_adjusted_sigma(sigma,N0,N):
    c = N/N0
    return np.sqrt(c)*sigma


def make_args(basic_args,alpha,N):
    a=10
    N0=10000
    new_args = copy.deepcopy(basic_args)
    new_args['alpha']=alpha
    new_args['cell_pop']=N
    Adj = basic_args['Adj']
    new_args['K'] = sim.make_K(new_args['alpha'],Adj)

    sigma = sigma_equiv(new_args['alpha'],a)
    print('sigma = ', sigma)
    sigma_adj = pop_adjusted_sigma(sigma,N0,new_args['cell_pop'])
    print('sigma_adj = ', sigma_adj)
    return new_args


#Generate a sequence of n sample times increasing according to given power
def generate_sample_times(T,power,n):
    a = T**(1/power)/(n-1)
    times = np.array([1+(a*k)**power for k in range(n)],dtype=int)
    return times



#**********************************
# MATCHING FUNCTIONS
#**********************************

#Get separation dependent correlation function from correlation matrix and distance matrix
def CFunc(C,dmat,bins):
    C1 = C.ravel()
    idxs = np.digitize(dmat.ravel(),bins = bins)
    Cs = [C1[idxs==i+1].mean() for i in range(len(bins))]
    return np.array(Cs)


#Calculate the matching matrix from a vector samples 
def match_matrix(sample):
    L = sample.shape[0]
    ones = np.ones(L)
    xx = np.outer(sample,ones)
    M = np.array(xx==xx.T,dtype=int)
    return M

#Calculate the matching probability matrix from a sample of probabilities
def match_prob_matrix(vs):
    L = vs.shape[0]
    ones = np.ones(L)
    vv = np.outer(vs,ones)
    vb = np.outer(1-vs,ones)
    M = np.array(vv*vv.T + vb*vb.T)
    return M

#Estimate the matching probability as a function of distance from a sample
def MFunc(state,dmat,bins):
    M=match_matrix(state)
    M1 = M.ravel()
    idxs = np.digitize(dmat.ravel(),bins = bins)
    Ms = [M1[idxs==i].mean() for i in range(1,len(bins))]
    return np.array(Ms)

#EStimate the matching probability as a function of distance from a sample of probabilties
def MProbFunc(vs,dmat,bins):
    M=match_prob_matrix(vs)
    M1 = M.ravel()
    idxs = np.digitize(dmat.ravel(),bins = bins)
    Ms = [M1[idxs==i].mean() for i in range(1,len(bins))]
    return np.array(Ms)

#Make the logarithmic neutral matching curve
def make_M_neut(epsilon):
    def f(r,b,c):
        a = b + c*np.log(np.where(r>epsilon,r/epsilon,1))
        return np.where(a>0.5,a,0.5)
    return f

#Interface matching function
def M_potts(r,mu):
    c = r/mu
    return np.exp(-c)*np.cosh(c)

#Interface correlation function
def C_potts(r,mu):
    return 2*M_potts(r,mu)-1

#**********************************
# PLOTTING CODE
#**********************************

def distribution_map(states,XUK,XB):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
    axes[0].scatter(XUK[:,0],XUK[:,1],c=states,s=20,cmap='rainbow',rasterized=True,vmin=0,vmax=1)
    axes[0].plot(XB[:,0],XB[:,1],'k',rasterized=True)
    axes[0].set_xlabel('Eastings (km)',size=16)
    axes[0].set_ylabel('Northings (km)',size=16)
    axes[0].tick_params(labelsize=14)
    bins = np.arange(0,1.01,0.05)
    axes[1].hist(states,color='orangered',edgecolor='k',linewidth=2,density=True,bins=bins,alpha=0.7,rasterized=True)
    axes[1].set_xlabel('v',size=16)
    axes[1].set_ylabel('P(v)',size=16)
    axes[1].tick_params(labelsize=14)
    plt.tight_layout()
    #plt.savefig(plots_folder/file,dpi=400)
    #plt.show()

def plot_sample(states,XS,XB):
    plt.figure(figsize=(4,5))
    plt.plot(XB[:,0],XB[:,1],'k',rasterized=True)
    plt.scatter(XS[:,0],XS[:,1],c=states,cmap='autumn',s=20,alpha=0.5,rasterized=True)
    plt.axis('off')
    plt.show()
    
    
#**********************************
# MORAN CODE
#**********************************

#Adjacency matrix for moran I
def make_Adj(XS,nn):
    nbrs = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(XS)
    distances, indices = nbrs.kneighbors(XS)
    L=len(XS)
    AI = np.zeros((L,L))
    for i in range(L):
        for j in indices[i]:
            AI[i,j]=1
    AI[np.diag_indices(L)]=0
    return AI
    

#Calculate Moran I from states using adjacency matrix A
def moran_I(states, A):
    mu = states.mean()
    var = states.var()
    if var<0.001:
        return 0.0
    dx = states-mu
    N = states.shape[0]
    ones = np.ones(N)
    xx = np.outer(ones,dx)
    dxdx = np.array(xx*xx.T)
    a = (A*dxdx).sum()
    b = (dx**2).sum()
    W = A.sum()
    return (N/W)*(a/b)   

def generate_moran_seqs(trials,args,A):
    I_seqs=[]
    for i in range(trials):
        samples, states  = sim.sample_sequence(args)
        Is=[]
        for sample,T in zip(samples,args['sample_times']):
            I = moran_I(sample,A)
            Is.append(I)
        I_seqs.append(Is)
    return np.array(I_seqs)
