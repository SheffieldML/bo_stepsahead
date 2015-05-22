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
objective_noisy = GPyOpt.fmodels.experiments2d.sixhumpcamel(sd = 0.15)     # noisy version
bounds = objective_noisy.bounds                                           # problem constrains 
input_dim = len(bounds)
f = objective_noisy.f 


## set initial points
X = samples_multidimensional_uniform(bounds,5)
Y = f(X)


max_iter = 20
# full msa
BO_msa_full = GPyOptmsa.msa.GPGOMSA(f,bounds, X,Y)       
BO_msa_full.run_optimization(max_iter=max_iter)

# 3 msa
BO_msa_3 = GPyOptmsa.msa.GPGOMSA(f,bounds, X,Y)       
BO_msa_3.run_optimization(max_iter=max_iter,n_ahead= 3)

# 5msa
BO_msa_5 = GPyOptmsa.msa.GPGOMSA(f,bounds, X,Y)       
BO_msa_5.run_optimization(max_iter=max_iter,n_ahead= 5)

BO_EL = GPyOpt.methods.BayesianOptimization(f=f,bounds=bounds,X=X,Y=Y,acquisition='EL1',)  
BO_EL.run_optimization(max_iter=max_iter, acqu_optimize_method = 'DIRECT')


idx = len(BO_msa_3.Y_best)
plt.plot(BO_msa_full.Y_best,label='EL-MSA-full')
plt.plot(BO_msa_3.Y_best,label='EL-MSA-3')
plt.plot(BO_msa_5.Y_best,label='EL-MSA-5')
plt.plot(BO_EL.Y_best,label='EL-standard')
plt.legend()
