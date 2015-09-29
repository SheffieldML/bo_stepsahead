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
experiments = [ 'func = cosines(sd=.0)',
                'func = branin(sd=.0)',
                'func = sixhumpcamel(sd=.0)',
                'func = mccormick(sd=.0)',
                'func = powers(sd=.0)',
                'func = eggholder(sd=.0)',
                'func = goldstein(sd=.0)',
                'func = alpine2(input_dim=2,sd=.0)'
                'func = alpine2(input_dim=5,sd=.0)',
                'func = alpine2(input_dim=10,sd=.0)'
                'func = gSobol(np.array([1,1]),sd=.1)',
                'func = gSobol(np.array([1,1,1,1,1]),sd=.1)',
                'func = gSobol(np.array([1,1,1,1,1,1,1,1,1,1]),sd=.1)'
              ]

NR              = 10            # Different initial points.
n_init          = 5             # number of initial points (per dimension).
max_iter_dim    = 10            # Number of iterations (per dimension).

for experiment in experiments:
    # --- problem setup
    exec experiment
    f         = func.f 
    bounds    = func.bounds
    input_dim = len(bounds)
    
    # --- Matrices to save results
    res_GLASSES_02    = np.empty([1,2])
    res_GLASSES_03    = np.empty([1,2])
    res_GLASSES_05    = np.empty([1,2])
    res_GLASSES_10    = np.empty([1,2])
    res_GLASSES_H    = np.empty([1,2])
    res_EL            = np.empty([1,2])

    # --- replicates 
    for k in range(NR):
        # --- inital points
        X = samples_multidimensional_uniform(bounds,n_init*len(bounds))
        Y = f(X)

        # --- Crete the optimization objects
        GLASSES_02 = GPyOptmsa.msa.GLASSES(f,bounds, X,Y,n_ahead=2) 
        GLASSES_03   = GPyOptmsa.msa.GLASSES(f,bounds, X,Y,n_ahead=3) 
        GLASSES_05   = GPyOptmsa.msa.GLASSES(f,bounds, X,Y,n_ahead=5)
        GLASSES_10   = GPyOptmsa.msa.GLASSES(f,bounds, X,Y,n_ahead=10)
        GLASSES_H    = GPyOptmsa.msa.GLASSES(f,bounds, X,Y)
        EL           = GPyOpt.methods.BayesianOptimization(f=f,bounds=bounds, X=X, Y=Y ,acquisition='EL') 

        # --- Run the optimizations
        print experiment
        print k
        print 'res_GLASSES_02'
        GLASSES_02.run_optimization(max_iter=max_iter_dim*input_dim)
        rep_col     = [k]*GLASSES_02.Y_best.shape[0]
        res_GLASSES_02 = np.vstack((res_GLASSES_02,np.vstack((rep_col,GLASSES_02.Y_best)).T))
        np.savetxt('res_GLASSES_02'+experiment[7:]+'.txt', res_GLASSES_02)

        print experiment
        print k
        print 'GLASSES_03'
        GLASSES_03.run_optimization(max_iter=max_iter_dim*input_dim) 
        rep_col     = [k]*GLASSES_03.Y_best.shape[0]
        res_GLASSES_03    = np.vstack((res_GLASSES_03,np.vstack((rep_col,GLASSES_03.Y_best)).T))
        np.savetxt('res_GLASSES_03_'+experiment[7:]+'.txt', res_GLASSES_03)
    
        print experiment
        print k
        print 'GLASSES_05'
        GLASSES_05.run_optimization(max_iter=max_iter_dim*input_dim)  
        rep_col     = [k]*GLASSES_05.Y_best.shape[0]
        res_GLASSES_05    = np.vstack((res_GLASSES_05,np.vstack((rep_col,GLASSES_05.Y_best)).T))
        np.savetxt('res_GLASSES_05_'+experiment[7:]+'.txt', res_GLASSES_05)

        print experiment
        print k
        print 'GLASSES_10'
        GLASSES_10.run_optimization(max_iter=max_iter_dim*input_dim)  
        rep_col     = [k]*GLASSES_10.Y_best.shape[0]
        res_GLASSES_10    = np.vstack((res_GLASSES_10,np.vstack((rep_col,GLASSES_10.Y_best)).T))
        np.savetxt('res_GLASSES_10_'+experiment[7:]+'.txt', res_GLASSES_10)

        print experiment
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

    
# -------- make comparison plot 
# choose experiment

n_exp = len(experiments)
methods = ['EL','GL-2','GL-3','GL-5','GL-10','GL-H']

results_mean = np.zeros((n_exp,len(methods)))
results_min = np.zeros((n_exp,len(methods)))
results_median = np.zeros((n_exp,len(methods)))


for k in range(len(experiments)): 
    results_mean[k,:] = create_table(experiments[k]).mean(1)
    results_min[k,:] = create_table(experiments[k]).min(1)
    results_median[k,:] = np.median(create_table(experiments[k]),1)


def create_table(experiment):    
    results = np.zeros((5,10))
    results[0,:] = value_at_last(np.loadtxt('res_EL_'+experiment[7:]+'.txt')[1:,:])
    results[1,:] = value_at_last(np.loadtxt('res_GLASSES_02_'+experiment[7:]+'.txt')[1:,:])
    results[2,:] = value_at_last(np.loadtxt('res_GLASSES_03_'+experiment[7:]+'.txt')[1:,:])
    results[3,:] = value_at_last(np.loadtxt('res_GLASSES_05_'+experiment[7:]+'.txt')[1:,:])
    results[4,:] = value_at_last(np.loadtxt('res_GLASSES_10_'+experiment[7:]+'.txt')[1:,:])
    results[5,:] = value_at_last(np.loadtxt('res_GLASSES_H_'+experiment[7:]+'.txt')[1:,:])
    return results

def value_at_last(data):
    data_val = data[:,1]
    data_id  = data[:,0]
    from itertools import groupby
    return np.array([data_val[data_id==i].min() for i, _ in groupby(data_id)])





















    
    
    
    








    
    
    
    
