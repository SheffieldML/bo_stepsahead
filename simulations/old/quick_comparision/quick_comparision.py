import GPyOptmsa
import GPyOpt 
import GPy
import numpy as np
from numpy.random import seed
from GPyOptmsa.util.general import samples_multidimensional_uniform, multigrid
from GPyOptmsa.util.acquisition import predict_locations
import matplotlib.pyplot as plt

seed(123)

# --- Objective function
objective_true  = GPyOpt.fmodels.experiments2d.sixhumpcamel()             # true function
objective_noisy = GPyOpt.fmodels.experiments2d.sixhumpcamel(sd = 0.1)     # noisy version
bounds = objective_noisy.bounds                                           # problem constrains 
input_dim = len(bounds)
f = objective_noisy.f 


## set initial points
X = samples_multidimensional_uniform(bounds,5)
Y = f(X)


# 3 msa
n_ahead = 3
max_iter = 10
BO_glasses = GPyOptmsa.msa.GLASSES(f,bounds, X,Y,n_ahead=n_ahead)       
BO_glasses.run_optimization(max_iter= max_iter, n_ahead=n_ahead)

# myopic
EL     = GPyOpt.methods.BayesianOptimization(f=f,bounds=bounds, X=X, Y=Y ,acquisition='EL1')
EL.run_optimization(max_iter= max_iter,acqu_optimize_method='DIRECT')