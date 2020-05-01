import numpy as np

class ising:
    def __init__(self,W,n_copies):
        self.W = W
        self.n_copies = n_copies
        self.L = self.W.shape[0]
        if self.W.shape[0] != self.W.shape[1]:
            print('W not square')
            
        self.W[np.diag_indices(self.L)]=0.0
        self.states = np.random.choice([-1,1],size=(self.n_copies,self.L))
                
    def update(self):
        idxs = np.random.randint(0,self.L,self.n_copies)
        Es = (self.W[idxs]*self.states).sum(axis=1)
        ps = 0.5*(1+np.tanh(Es))
        us = np.random.uniform(0,1,self.n_copies)
        new_states = np.where(us<ps,1,-1)
        for i in range(self.n_copies):
            self.states[i,idxs[i]]=new_states[i]
    
    def calibrate(self,C_target,iterations, learning_rate, update_loops):
        self.W = np.zeros(shape=(self.L,self.L))
        for i in range(iterations):
            for j in range(update_loops):
                self.update()
            C = self.get_C()
            dC = C_target-C
            self.W += learning_rate*dC
            self.W[np.diag_indices(self.L)]=0.0
            
            dCAbs = np.abs(dC)
            a = np.round(dCAbs.mean(),4)
            b = np.round(np.amax(dCAbs),4)
            print(i, 'dC_mean = ',a,' dC max = ', b,end='\r')         
                       
    def get_C(self):
        ones = np.ones(self.L)
        C = np.zeros((self.L,self.L))
        for x in self.states:
            xx = np.outer(x,ones)
            C += np.array(xx*xx.T)
        return C/self.n_copies
    
    def estimate_lnZ(self,W_target,K,I):
        self.L = W_target.shape[0]
        self.states = np.random.choice([-1,1],size=(self.n_copies,self.L))
        
        E_sums = np.zeros(self.n_copies)
        
        for k in range(K):
            self.W = (k/K)*W_target
            self.W[np.diag_indices(self.L)]=0.0
            
            for i in range(I):
                self.update()
            
            ones = np.ones(self.L)
            for i,x in enumerate(self.states):
                xx = np.outer(x,ones)
                E_sums[i] += 0.5*(np.array(xx*xx.T)*W_target).sum()

        
        self.lnZ = self.L*np.log(2.0) + np.log(np.exp(E_sums/K).mean())
        return self.lnZ
            

        
def calib(cargs):
    n_copies = cargs['n_copies']
    iterations = cargs['iterations'] 
    learning_rate = cargs['learning_rate']
    update_loops = cargs['update_loops']
    C = cargs['C']
    W = np.zeros(C.shape)
    
    sys = ising(W,n_copies)
    sys.calibrate(C,iterations,learning_rate,update_loops)
    return sys.W, sys.get_C()        
        
        
        
def correl_matrix(samples):
    n,L = samples.shape
    C = np.zeros((L,L))
    ones = np.ones(L)
    for x in samples:
        xx = np.outer(x,ones)
        C += np.array(xx*xx.T)
    return C/n


def lnZ(zargs):
    n_copies = zargs['n_copies']
    K = zargs['anneal_steps'] 
    I = zargs['intermediate_steps']
    theta = zargs['theta']
    W = np.zeros(theta.shape)
    
    sys = ising(W,n_copies)
    lnZ = sys.estimate_lnZ(theta,K,I)
    print('ln Z = ', lnZ)
    return lnZ      


def lnP(states,theta,lnZ):
    L = theta.shape[0]
    ones = np.ones(L)
    xx = np.outer(states,ones)
    E = 0.5*(np.array(xx*xx.T)*theta).sum()
    return E - lnZ


def E(states,theta):
    L = theta.shape[0]
    ones = np.ones(L)
    xx = np.outer(states,ones)
    E = 0.5*(np.array(xx*xx.T)*theta).sum()
    return E
    
        
        

        