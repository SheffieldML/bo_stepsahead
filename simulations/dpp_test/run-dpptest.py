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


max_iter_dim = 1
n_init       = 25
func         = cosines(sd=.1)
f            = func.f 
bounds       = func.bounds
input_dim    = len(bounds)


res_HBO5     = np.empty([1,2])
X            = samples_multidimensional_uniform(bounds,n_init*len(bounds))
Y            = f(X)

HBO5         = GPyOptmsa.msa.GPGOMSA(f,bounds, X,Y)


HBO5.run_optimization(max_iter=1,n_ahead=3)  








    
    
    
    








    
    
    
    
