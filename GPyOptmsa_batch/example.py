import GPyOptmsa
import GPyOpt 
import GPy
import numpy as np
from numpy.random import seed
from GPyOptmsa.util.general import samples_multidimensional_uniform, multigrid
import matplotlib.pyplot as plt

seed(123)

# --- Objective function
objective_true  = GPyOpt.fmodels.experiments2d.sixhumpcamel()             # true function
objective_noisy = GPyOpt.fmodels.experiments2d.sixhumpcamel(sd = 0.1)     # noisy version
bounds = objective_noisy.bounds                                           # problem constrains 
input_dim = len(bounds)

f = objective_noisy.f 

X = samples_multidimensional_uniform(bounds,20)
Y = f(X)

# Crete the optimization object
BO = GPyOptmsa.msa.GPGOMSA(f,bounds, X,Y)       
    
# Runs optimization
BO.run_optimization(max_iter=3,beta=10)

# Plot the acquisition
BO.plot_loss()

# Plot evaluation
BO.plot_convergence()







