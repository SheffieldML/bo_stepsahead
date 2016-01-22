import GPyOptmsa
import GPyOpt
import GPy
from numpy.random import seed
import numpy as np
from GPyOptmsa.util.general import samples_multidimensional_uniform, multigrid

import warnings
warnings.filterwarnings("ignore")
seed(12345)

from GPyOpt.fmodels.experiments2d import *
from GPyOpt.fmodels.experimentsNd import * 

 
## Simulation setup    
experiments = ['func = cosines(sd=.1)'
              ]

NR              = 5            # Different initial points.
n_init          = 5             # number of initial points (per dimension).
max_iter_dim    = 10            # Number of iterations (per dimension).

for experiment in experiments:
    # --- problem setup
    exec experiment
    f         = func.f 
    bounds    = func.bounds
    input_dim = len(bounds)
    
    # --- Matrices to save results
    res_GLASSES_H    = np.empty([1,2])
    res_EL           = np.empty([1,2])
    res_LCB          = np.empty([1,2])
    res_MPI          = np.empty([1,2])


    # --- replicates 
    for k in range(NR):
        # --- inital points
        X = samples_multidimensional_uniform(bounds,n_init*len(bounds))
        Y = f(X)

        # --- Crete the optimization objects
        GLASSES_H    = GPyOptmsa.msa.GLASSES(f,bounds, X,Y)
        EL           = GPyOpt.methods.BayesianOptimization(f=f,bounds=bounds, X=X, Y=Y ,acquisition='EL') 
        LCB          = GPyOpt.methods.BayesianOptimization(f=f,bounds=bounds, X=X, Y=Y ,acquisition='LCB',acquisition_par=2) 
        MPI          = GPyOpt.methods.BayesianOptimization(f=f,bounds=bounds, X=X, Y=Y ,acquisition='MPI') 

        # --- Run the optimizations
        #print experiment
        print k
        print 'GLASSES_H'
        GLASSES_H.run_optimization(max_iter=max_iter_dim*input_dim,ahead_remaining = True)  
        rep_col     = [k]*GLASSES_H.Y_best.shape[0]
        res_GLASSES_H    = np.vstack((res_GLASSES_H,np.vstack((rep_col,GLASSES_H.Y_best)).T))
        np.savetxt('res_GLASSES_H_'+experiment[7:]+'.txt', res_GLASSES_H)

        print experiment
        print k
        print 'EL'
        EL.run_optimization(max_iter=max_iter_dim*input_dim, acqu_optimize_method='DIRECT')  
        rep_col     = [k]*EL.Y_best.shape[0]
        res_EL    = np.vstack((res_EL,np.vstack((rep_col,EL.Y_best)).T))
        np.savetxt('res_EL_'+experiment[7:]+'.txt', res_EL)

        print experiment
        print k
        print 'LCB'
        LCB.run_optimization(max_iter=max_iter_dim*input_dim, acqu_optimize_method='DIRECT')  
        rep_col     = [k]*LCB.Y_best.shape[0]
        res_LCB    = np.vstack((res_LCB,np.vstack((rep_col,LCB.Y_best)).T))
        np.savetxt('res_LCB_'+experiment[7:]+'.txt', res_LCB)

        print experiment
        print k
        print 'MPI'
        MPI.run_optimization(max_iter=max_iter_dim*input_dim, acqu_optimize_method='DIRECT')  
        rep_col     = [k]*MPI.Y_best.shape[0]
        res_MPI    = np.vstack((res_MPI,np.vstack((rep_col,MPI.Y_best)).T))
        np.savetxt('res_MPI_'+experiment[7:]+'.txt', res_MPI)


## we make the comparative plot here


def get_mean(data_set):
    mean_set = np.array([0]*10)
    for k in range(10):
        mean_set =+ data_set[data_set[:,0]==k][:,1]
    return mean_set/10


data_glasses = np.loadtxt('res_GLASSES_H_'+experiment[7:]+'.txt')
data_EL = np.loadtxt('res_EL_'+experiment[7:]+'.txt')
data_MPI = np.loadtxt('res_MPI_'+experiment[7:]+'.txt')
data_LCB = np.loadtxt('res_LCB_'+experiment[7:]+'.txt')

plt.plot(get_mean(data_glasses),label)
plt.plot(get_mean(data_EL))
plt.plot(get_mean(data_MPI))
plt.plot(get_mean(data_LCB))
















    
    
    
    








    
    
    
    
