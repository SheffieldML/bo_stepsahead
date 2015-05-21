import GPyOptmsa
import GPyOpt
import GPy
from numpy.random import seed
import numpy as np
import pandas as pd
from GPyOptmsa.util.general import samples_multidimensional_uniform, multigri

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
                'func = alpine1(input_dim=10,sd=.15)',
                'func = alpine1(input_dim=20,sd=.15)']

NR              = 10
n_init          = 4
max_iter_dim    = 15

for experiment in experiments:
    # --- problem setup
    exec experiment
    f         = func.f 
    bounds    = func.bounds
    input_dim = len(bounds)
    
    # --- Matrices to save results
    res_HBOfull = []
    res_HBO3    = []
    res_HBO5    = []
    res_HBO10   = []
    res_HBO15   = []
    res_BOEL    = []

    # --- replicates 
    for k in range(NR):
        # --- inital points
        init_points = samples_multidimensional_uniform(bounds,n_init*len(bounds))
        Y = f(init_points)

        # --- Crete the optimization objects
        HBOfull = GPyOptmsa.msa.GPGOMSA(f,bounds, X,Y) 
        HBO3    = GPyOptmsa.msa.GPGOMSA(f,bounds, X,Y) 
        HBO5    = GPyOptmsa.msa.GPGOMSA(f,bounds, X,Y)
        HBO10   = GPyOptmsa.msa.GPGOMSA(f,bounds, X,Y)
        HBO15   = GPyOptmsa.msa.GPGOMSA(f,bounds, X,Y) 
        BOEL    = GPyOpt.methods.BayesianOptimization(f=f,bounds=bounds, X=X,Y=Y) 

        # --- Run the optimizations
        HBOfull.run_optimization(max_iter=max_iter_dim*input_dim)
        rep_col     = [k]*HBOfull.Y_best.shape[0]
        res_HBOfull = np.vstack((res_HBOfull,np.hstack((rep_col,res_HBOfull.Y_best.shape))))
        np.save('res_HBOfull_'+experiment[7:]+'.txt', res_HBOfull)

        HBO3.run_optimization(max_iter=max_iter_dim*input_dim,n_ahead=3) 
        rep_col     = [k]*HBO3.Y_best.shape[0]
        res_HBO3    = np.vstack((res_HBO3,np.hstack((rep_col,res_HBO3.Y_best.shape))))
        np.save('res_HBO3_'+experiment[7:]+'.txt', res_HBO3)
    
        HBO5.run_optimization(max_iter=max_iter_dim*input_dim,n_ahead=5)  
        rep_col     = [k]*HBO5.Y_best.shape[0]
        res_HBO5    = np.vstack((res_HBO5,np.hstack((rep_col,res_HBO5.Y_best.shape))))
        np.save('res_HBO5_'+experiment[7:]+'.txt', res_HBO5)
        
        HBO10.run_optimization(max_iter=max_iter_dim*input_dim,n_ahead=10) 
        rep_col     = [k]*HBO10.Y_best.shape[0]
        res_HBO10   = np.vstack((res_HBO10,np.hstack((rep_col,res_HBO10.Y_best.shape))))  
        np.save('res_HBO10_'+experiment[7:]+'.txt', res_HBO10)    

        HBO15.run_optimization(max_iter=max_iter_dim*input_dim,n_ahead=15)   
        rep_col     = [k]*HBO15.Y_best.shape[0]
        res_HB15    = np.vstack((res_HBO15,np.hstack((rep_col,res_HBO15.Y_best.shape))))   
        np.save('res_HBO15_'+experiment[7:]+'.txt', res_HBO15)     

        BOEL.run_optimization(max_iter=max_iter_dim*input_dim,acqu_optimize_method=='DIRECT')  
        rep_col     = [k]*BOEL.Y_best.shape[0]
        res_BOEL    = np.vstack((res_BOEL,np.hstack((rep_col,res_BOEL.Y_best.shape))))
        np.save('res_BOEL_'+experiment[7:]+'.txt', res_BOEL)

    
    
    
    
    
    








    
    
    
    
