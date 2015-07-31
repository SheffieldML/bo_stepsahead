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
experiments = [ 'func = cosines(sd=.1)',
                'func = branin(sd=.1)',
#                'func = goldstein(sd=.1)',
#                'func = sixhumpcamel(sd=.1)',
#                'func = mccormick(sd=.1)',
#                'func = powers(sd=.1)',
#                'func = alpine1(input_dim=2,sd=.1)',
#                'func = alpine1(input_dim=5,sd=.1)',
#                'func = alpine1(input_dim=10,sd=.1)'
              ]

NR              = 5             # Different initial points.
n_init          = 5             # number of initial points (per dimension).
max_iter_dim    = 10            # Number of iterations (per dimension).

experiment = experiments[0]
for experiment in experiments:
    # --- problem setup
    exec experiment
    f         = func.f 
    bounds    = func.bounds
    input_dim = len(bounds)
    
    # --- Matrices to save results
    res_GLASSES_full  = np.empty([1,2])
    res_GLASSES_O3    = np.empty([1,2])
    res_GLASSES_O5    = np.empty([1,2])
    res_EL            = np.empty([1,2])

    # --- replicates 
    for k in range(NR):
        # --- inital points
        X = samples_multidimensional_uniform(bounds,n_init*len(bounds))
        Y = f(X)

        # --- Crete the optimization objects
        GLASSES_full = GPyOptmsa.msa.GLASSES(f,bounds, X,Y) 
        GLASSES_O3   = GPyOptmsa.msa.GLASSES(f,bounds, X,Y) 
        GLASSES_O5   = GPyOptmsa.msa.GLASSES(f,bounds, X,Y)
        EL           = GPyOpt.methods.BayesianOptimization(f=f,bounds=bounds, X=X, Y=Y ,acquisition='EL1') 

        # --- Run the optimizations
        print experiment
        print k
        print 'res_GLASSES_full'
        GLASSES_full.run_optimization(max_iter=max_iter_dim*input_dim,n_ahead=2)
        rep_col     = [k]*GLASSES_full.Y_best.shape[0]
        res_GLASSES_full = np.vstack((res_GLASSES_full,np.vstack((rep_col,GLASSES_full.Y_best)).T))
        np.savetxt('res_GLASSES_full'+experiment[7:]+'.txt', res_GLASSES_full)

        print experiment
        print k
        print 'GLASSES_O3'
        GLASSES_O3.run_optimization(max_iter=max_iter_dim*input_dim,n_ahead=3) 
        rep_col     = [k]*GLASSES_O3.Y_best.shape[0]
        res_GLASSES_O3    = np.vstack((res_GLASSES_O3,np.vstack((rep_col,GLASSES_O3.Y_best)).T))
        np.savetxt('res_GLASSES_O3_'+experiment[7:]+'.txt', res_GLASSES_O3)
    
        print experiment
        print k
        print 'GLASSES_O5'
        GLASSES_O5.run_optimization(max_iter=max_iter_dim*input_dim,n_ahead=5)  
        rep_col     = [k]*GLASSES_O5.Y_best.shape[0]
        res_GLASSES_O5    = np.vstack((res_GLASSES_O5,np.vstack((rep_col,GLASSES_O5.Y_best)).T))
        np.savetxt('res_GLASSES_O5_'+experiment[7:]+'.txt', res_GLASSES_O5)

        print experiment
        print k
        print 'EL'
        EL.run_optimization(max_iter=max_iter_dim*input_dim, acqu_optimize_method='DIRECT')  
        rep_col     = [k]*EL.Y_best.shape[0]
        res_EL    = np.vstack((res_EL,np.vstack((rep_col,EL.Y_best)).T))
        np.savetxt('res_EL_'+experiment[7:]+'.txt', res_EL)

    
# -------- make comparison plot 
# choose experiment
experiment = experiments[0] 
n_inits      = 26 
n_replicates = 4

# -------- load results
best_res_GLASSES_full   = np.loadtxt('res_HBOfull_'+experiment[7:]+'.txt')[1:(n_inits*n_replicates+1),1].reshape(n_replicates,n_inits).mean(0)
best_GLASSES_O3       = np.loadtxt('res_GLASSES_O3_'+experiment[7:]+'.txt')[1:(n_inits*n_replicates+1),1].reshape(n_replicates,n_inits).mean(0)
best_GLASSES_O5       = np.loadtxt('res_GLASSES_O5_'+experiment[7:]+'.txt')[1:(n_inits*n_replicates+1),1].reshape(n_replicates,n_inits).mean(0)
best_HBO10      = np.loadtxt('res_HBO10_'+experiment[7:]+'.txt')[1:(n_inits*n_replicates+1),1].reshape(n_replicates,n_inits).mean(0)
best_EL       = np.loadtxt('res_EL_'+experiment[7:]+'.txt')[1:(n_inits*n_replicates+1),1].reshape(n_replicates,n_inits).mean(0)

plt.plot(best_HBOfull,label='full GLASSES')
plt.plot(best_GLASSES_O3 ,label='GLASSES-3-steps')
plt.plot(best_GLASSES_O5,label='GLASSES-5-steps')
plt.plot(best_EL,'b-.',label='myopic-EL')
plt.xlabel('iteration')
plt.ylabel('Best value')

plt.legend()
















    
    
    
    








    
    
    
    
