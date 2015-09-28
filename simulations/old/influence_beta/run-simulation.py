import GPyOptmsa
import GPyOpt
import GPy
from numpy.random import seed
import numpy as np
import pandas as pd
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

NR              = 20
n_init          = 4
max_iter_dim    = 10

for experiment in experiments:
    # --- problem setup
    exec experiment
    f         = func.f 
    bounds    = func.bounds
    input_dim = len(bounds)
    
    # --- Matrices to save results
    res_HBObeta0     = np.empty([1,2])
    res_HBObeta01    = np.empty([1,2])
    res_HBObeta1     = np.empty([1,2])
    res_HBObeta5     = np.empty([1,2])
    res_HBObeta10    = np.empty([1,2])

    # --- replicates 
    for k in range(NR):
        # --- inital points
        X = samples_multidimensional_uniform(bounds,n_init*len(bounds))
        Y = f(X)

        # --- Crete the optimization objects
        HBObeta0   = GPyOptmsa.msa.GPGOMSA(f,bounds, X,Y) 
        HBObeta01  = GPyOptmsa.msa.GPGOMSA(f,bounds, X,Y) 
        HBObeta1   = GPyOptmsa.msa.GPGOMSA(f,bounds, X,Y)
        HBObeta5   = GPyOptmsa.msa.GPGOMSA(f,bounds, X,Y)
        HBObeta10  = GPyOptmsa.msa.GPGOMSA(f,bounds, X,Y) 

        # --- Run the optimizations
        print experiment
        print k
        print 'HBObeta0'
        HBObeta0.run_optimization(max_iter=max_iter_dim*input_dim)
        rep_col     = [k]*HBObeta0.Y_best.shape[0]
        res_HBObeta0 = np.vstack((res_HBObeta0,np.vstack((rep_col,HBObeta0.Y_best)).T))
        np.savetxt('res_HBObeta0_'+experiment[7:]+'.txt', res_HBObeta0)

        print experiment
        print k
        print 'HBObeta01'
        HBObeta01.run_optimization(max_iter=max_iter_dim*input_dim,beta=0.1) 
        rep_col     = [k]*HBObeta01.Y_best.shape[0]
        res_HBObeta01    = np.vstack((res_HBObeta01,np.vstack((rep_col,HBObeta01.Y_best)).T))
        np.savetxt('res_HBO3_'+experiment[7:]+'.txt', res_HBObeta01)
    
        print experiment
        print k
        print 'HBObeta1'
        HBObeta1.run_optimization(max_iter=max_iter_dim*input_dim,beta=1)  
        rep_col     = [k]*HBObeta1.Y_best.shape[0]
        res_HBObeta1   = np.vstack((res_HBObeta1,np.vstack((rep_col,HBObeta1.Y_best)).T))
        np.savetxt('res_HBObeta1_'+experiment[7:]+'.txt', res_HBObeta1)
        
        print experiment
        print k
        print 'HBObeta5'
        HBObeta5.run_optimization(max_iter=max_iter_dim*input_dim, beta=5) 
        rep_col     = [k]*HBObeta5.Y_best.shape[0]
        res_HBObeta5   = np.vstack((res_HBObeta5,np.vstack((rep_col,HBObeta5.Y_best)).T))  
        np.savetxt('res_HBObeta5_'+experiment[7:]+'.txt', res_HBObeta5)    

        print experiment
        print k
        print 'HBObeta10'
        HBObeta10.run_optimization(max_iter=max_iter_dim*input_dim, beta=10)   
        rep_col     = [k]*HBObeta10.Y_best.shape[0]
        res_HBObeta10    = np.vstack((res_HBObeta10,np.vstack((rep_col,HBObeta10.Y_best)).T))   
        np.savetxt('res_HBObeta10_'+experiment[7:]+'.txt', res_HBObeta10)     


    
    
    
    
    








    
    
    
    
