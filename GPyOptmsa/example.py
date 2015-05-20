import GPyOptmsa
import GPy
import numpy as np
from numpy.random import seed
from GPyOptmsa.util.general import samples_multidimensional_uniform, multigrid
import matplotlib.pyplot as plt

seed(123)


## Problem definition    
bounds = [(-0.5*np.pi,0.5*np.pi),(-0.5*np.pi,0.5*np.pi)]

def f(x):
    x1 = x[:,0]
    x2 = x[:,1]
    y  = np.sin(x1)*np.sin(x2)
    return  y.reshape(len(y),1)

                                        # problem constrains 
input_dim = len(bounds)

X = samples_multidimensional_uniform(bounds,50)
Y = f(X)

# --- Crete the object
BO = GPyOptmsa.msa.GPGOMSA(f,bounds, X,Y)       
    
# Runs optimization
BO.run_optimization(5)











