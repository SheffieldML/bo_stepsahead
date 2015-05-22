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
experiments = [ 'func = cosines(sd=.15)',
                'func = branin(sd=.15)',
                'func = goldstein(sd=.15)',
                'func = sixhumpcamel(sd=.15)',
                'func = mccormick(sd=.15)',
                'func = powers(sd=.15)',
                'func = alpine1(input_dim=2,sd=.15)',
                'func = alpine1(input_dim=5,sd=.15)',
                'func = alpine1(input_dim=10,sd=.15)']

NR              = 20            # Different initial points.
n_init          = 4             # number of initial points (per dimension).
max_iter_dim    = 10            # Number of iterations (per dimension).

for experiment in experiments:
    # --- problem setup
    exec experiment
    f         = func.f 
    bounds    = func.bounds
    input_dim = len(bounds)
    
    # --- Matrices to save results
    res_HBOfull = np.empty([1,2])
    res_HBO3    = np.empty([1,2])
    res_HBO5    = np.empty([1,2])
    res_HBO10   = np.empty([1,2])
    res_HBO15   = np.empty([1,2])
    res_BOEL    = np.empty([1,2])

    # --- replicates 
    for k in range(NR):
        # --- inital points
        X = samples_multidimensional_uniform(bounds,n_init*len(bounds))
        Y = f(X)

        # --- Crete the optimization objects
        HBOfull = GPyOptmsa.msa.GPGOMSA(f,bounds, X,Y) 
        HBO3    = GPyOptmsa.msa.GPGOMSA(f,bounds, X,Y) 
        HBO5    = GPyOptmsa.msa.GPGOMSA(f,bounds, X,Y)
        HBO10   = GPyOptmsa.msa.GPGOMSA(f,bounds, X,Y)
        HBO15   = GPyOptmsa.msa.GPGOMSA(f,bounds, X,Y) 
        BOEL    = GPyOpt.methods.BayesianOptimization(f=f,bounds=bounds, X=X,Y=Y) 


        # --- Run the optimizations
        print experiment
        print k
        print 'HBOfull'
        HBOfull.run_optimization(max_iter=max_iter_dim*input_dim)
        rep_col     = [k]*HBOfull.Y_best.shape[0]
        res_HBOfull = np.vstack((res_HBOfull,np.vstack((rep_col,HBOfull.Y_best)).T))
        np.savetxt('res_HBOfull_'+experiment[7:]+'.txt', res_HBOfull)

        print experiment
        print k
        print 'HBO3'
        HBO3.run_optimization(max_iter=max_iter_dim*input_dim,n_ahead=3) 
        rep_col     = [k]*HBO3.Y_best.shape[0]
        res_HBO3    = np.vstack((res_HBO3,np.vstack((rep_col,HBO3.Y_best)).T))
        np.savetxt('res_HBO3_'+experiment[7:]+'.txt', res_HBO3)
    
        print experiment
        print k
        print 'HBO5'
        HBO5.run_optimization(max_iter=max_iter_dim*input_dim,n_ahead=5)  
        rep_col     = [k]*HBO5.Y_best.shape[0]
        res_HBO5    = np.vstack((res_HBO5,np.vstack((rep_col,HBO5.Y_best)).T))
        np.savetxt('res_HBO5_'+experiment[7:]+'.txt', res_HBO5)
 
        print experiment
        print k       
        print 'HBO10'
        HBO10.run_optimization(max_iter=max_iter_dim*input_dim,n_ahead=10) 
        rep_col     = [k]*HBO10.Y_best.shape[0]
        res_HBO10   = np.vstack((res_HBO10,np.vstack((rep_col,HBO10.Y_best)).T))  
        np.savetxt('res_HBO10_'+experiment[7:]+'.txt', res_HBO10)    

        print experiment
        print k
        print 'HBO15'
        HBO15.run_optimization(max_iter=max_iter_dim*input_dim,n_ahead=15)   
        rep_col     = [k]*HBO15.Y_best.shape[0]
        res_HB15    = np.vstack((res_HBO15,np.vstack((rep_col,HBO15.Y_best)).T))   
        np.save('res_HBO15_'+experiment[7:]+'.txt', res_HBO15)     

        print experiment
        print k
        print 'BOEL'
        BOEL.run_optimization(max_iter=max_iter_dim*input_dim,acqu_optimize_method='DIRECT')  
        rep_col     = [k]*BOEL.Y_best.shape[0]
        res_BOEL    = np.vstack((res_BOEL,np.vstack((rep_col,BOEL.Y_best)).T))
        np.savetxt('res_BOEL_'+experiment[7:]+'.txt', res_BOEL)

    
    
    
    
    








    
    
    
    
