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
    res_HBOdpp1     = np.empty([1,2])
    res_HBOdpp5     = np.empty([1,2])
    res_HBOdpp10    = np.empty([1,2])
    res_HBOdpp20    = np.empty([1,2])

    # --- replicates 
    for k in range(NR):
        # --- inital points
        X = samples_multidimensional_uniform(bounds,n_init*len(bounds))
        Y = f(X)

        # --- Crete the optimization objects
        HBOdpp1    = GPyOptmsa.msa.GPGOMSA(f,bounds, X,Y) 
        HBOdpp5    = GPyOptmsa.msa.GPGOMSA(f,bounds, X,Y) 
        HBOdpp10   = GPyOptmsa.msa.GPGOMSA(f,bounds, X,Y)
        HBOdpp20   = GPyOptmsa.msa.GPGOMSA(f,bounds, X,Y)
        

        # --- Run the optimizations
        print experiment
        print k
        print 'HBOdpp1'
        HBOdpp1.run_optimization(max_iter=max_iter_dim*input_dim,n_samples_dpp=1)
        rep_col         = [k]*HBOdpp1.Y_best.shape[0]
        res_HBOdpp1     = np.vstack((res_HBOdpp1,np.vstack((rep_col,HBOdpp1.Y_best)).T))
        np.savetxt('res_HBObeta0_'+experiment[7:]+'.txt', res_HBOdpp1)

        print experiment
        print k
        print 'HBOdpp5'
        HBOdpp5.run_optimization(max_iter=max_iter_dim*input_dim,n_samples_dpp=5) 
        rep_col         = [k]*HBOdpp5.Y_best.shape[0]
        res_HBOdpp5     = np.vstack((res_HBOdpp5,np.vstack((rep_col,HBOdpp5.Y_best)).T))
        np.savetxt('res_HBO3_'+experiment[7:]+'.txt', res_HBOdpp5)
    
        print experiment
        print k
        print 'HBOdpp10'
        HBOdpp10.run_optimization(max_iter=max_iter_dim*input_dim,n_samples_dpp=10)  
        rep_col         = [k]*HBOdpp10.Y_best.shape[0]
        res_HBOdpp10    = np.vstack((res_HBOdpp10,np.vstack((rep_col,HBOdpp10.Y_best)).T))
        np.savetxt('res_HBObeta1_'+experiment[7:]+'.txt', res_HBOdpp10)
        
        print experiment
        print k
        print 'HBOdpp20'
        HBOdpp20.run_optimization(max_iter=max_iter_dim*input_dim, n_samples_dpp=20) 
        rep_col         = [k]*HBOdpp20.Y_best.shape[0]
        res_HBOdpp20    = np.vstack((res_HBOdpp20,np.vstack((rep_col,HBOdpp20.Y_best)).T))  
        np.savetxt('res_HBObeta5_'+experiment[7:]+'.txt', res_HBOdpp20)    

    
    
    
    









    
    
    
    
