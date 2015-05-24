import GPyOptmsa
from GPyOptmsa.util.general import samples_multidimensional_uniform, multigrid
import GPyOpt
import GPy
from numpy.random import seed
import numpy as np

# Iport roBO features
from robo.models.GPyModel import GPyModel
from robo.acquisition.EI import EI
from robo.acquisition.Entropy import Entropy
from robo.maximizers.maximize import DIRECT
from robo.recommendation.incumbent import compute_incumbent
from robo import BayesianOptimization
seed(12345)

# import problems
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
    X_lower   = np.asarray(bounds)[:,0]
    X_upper   = np.asarray(bounds)[:,1]
    input_dim = len(bounds)
    
    # --- Matrices to save results
    res_HBO     = np.empty([1,2])
    res_ES    = np.empty([1,2])
    res_EI    = np.empty([1,2])

    # --- replicates 
    for k in range(NR):
        # --- inital points
        X = samples_multidimensional_uniform(bounds,n_init*len(bounds))
        Y = f(X)

        # --- Crete the optimization objects
        # Holistic BO
        HBO     = GPyOptmsa.msa.GPGOMSA(f=f,bounds= bounds, X=X,Y=Y) 
        
        # Entropy Search
        kernel                  = GPy.kern.RBF(input_dim, variance=.1, lengthscale=.1)  + GPy.kern.Bias(input_dim)
        model                   = GPyModel(kernel, optimize=True, noise_variance=1e-4, num_restarts=10)
        proposal_measurement    = EI(model, X_upper=X_upper, X_lower=X_lower, compute_incumbent=compute_incumbent, par=0.1)
        acquisition_func        = Entropy(model, X_lower, X_upper, sampling_acquisition=proposal_measurement)
        maximizer               = DIRECT

        ES = BayesianOptimization(  acquisition_fkt = acquisition_func,
                                    model           = model,
                                    maximize_fkt    = maximizer,
                                    X_lower         = X_lower,
                                    X_upper         = X_upper,
                                    dims            = input_dim,
                                    objective_fkt   = f)

        # Expected Improvement
        EI      = GPyOpt.methods.BayesianOptimization(f=f,bounds=bounds, X=X, Y=Y)

        ## NOTES: Need to add wall-clock time for the tre methods
        # time = time.time()



        # --- Run the optimizations
        print experiment
        print k
        print 'HBO'
        HBO.run_optimization(max_iter=max_iter_dim*input_dim)
        rep_col     = [k]*HBO.Y_best.shape[0]
        res_HBO = np.vstack((res_HBO,np.vstack((rep_col,HBO.Y_best)).T))
        np.savetxt('res_HBO_'+experiment[7:]+'.txt', res_HBO)

        print experiment
        print k
        print 'ES'
        ES.run_optimization(num_iterations=max_iter_dim*input_dim) 
        #rep_col     = [k]*HBO3.Y_best.shape[0]
        #res_HBO3    = np.vstack((res_HBO3,np.vstack((rep_col,HBO3.Y_best)).T))
        np.savetxt('res_EI_'+experiment[7:]+'.txt', res_EI)
    
        print experiment
        print k
        print 'EI'
        EI.run_optimization(max_iter=max_iter_dim*input_dim)  
        rep_col     = [k]*EI.Y_best.shape[0]
        res_EI      = np.vstack((res_EI,np.vstack((rep_col,EI.Y_best)).T))
        np.savetxt('res_EI_'+experiment[7:]+'.txt', res_EI)
 

    
    
    








    
    
    
    
