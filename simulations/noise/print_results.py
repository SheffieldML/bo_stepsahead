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
experiments = [ 'func = cosines(sd=.0)',
                'func = cosines(sd=.1)',
                'func = cosines(sd=.25)']
    
# -------- make comparison plot 
# choose experiment

n_exp = len(experiments)
methods = ['EL','GL-H']

results_mean = np.zeros((n_exp,len(methods)))
results_min = np.zeros((n_exp,len(methods)))
results_median = np.zeros((n_exp,len(methods)))


for k in range(len(experiments)): 
    results_mean[k,:] = create_table(experiments[k]).mean(1)
    results_min[k,:] = create_table(experiments[k]).min(1)
    results_median[k,:] = np.median(create_table(experiments[k]),1)


results_mean.round(4)


experiment = experiments[0]



def create_table_best(experiment):    
    # get the best value of each run
    results_last = np.zeros((2,5))
    results_last[0,:] = value_at_last(np.loadtxt('res_EL_'+experiment[7:]+'.txt')[1:,:])
    results_last[1,:] = value_at_last(np.loadtxt('res_GLASSES_H_'+experiment[7:]+'.txt')[1:,:])

    return results_last




def create_table(experiment):    
    
    # get the best value of each run
    results_last = np.zeros((2,5))
    results_last[0,:] = value_at_last(np.loadtxt('res_EL_'+experiment[7:]+'.txt')[1:,:])
    results_last[1,:] = value_at_last(np.loadtxt('res_GLASSES_H_'+experiment[7:]+'.txt')[1:,:])

    # get the initial value of each run
    results_first = np.zeros((2,5))
    results_first[0,:] = value_at_first(np.loadtxt('res_EL_'+experiment[7:]+'.txt')[1:,:])
    results_first[1,:] = value_at_first(np.loadtxt('res_GLASSES_H_'+experiment[7:]+'.txt')[1:,:])

    # get the optimum
    exec experiment
    best = func.fmin
    gap = (results_first-results_last)/(results_first-best)
    
    # normalization due to noise
    #gap = (gap + gap.min()) / np.max((gap + gap.min()))
    #for row in xrange(gap.shape[0]):
    #    gap[row] = (gap[row] + np.min(gap[row])) / np.max((gap[row] + np.min(gap[row])))

    return gap




def value_at_last(data):
    idx = data[:,0]<5
    data_val = data[idx,1]
    data_id  = data[idx,0]
    from itertools import groupby
    return np.array([data_val[data_id==i].min() for i, _ in groupby(data_id)])

def value_at_first(data):
    idx = data[:,0]<5
    data_val = data[idx,1]
    data_id  = data[idx,0]
    from itertools import groupby
    return np.array([data_val[data_id==i].max() for i, _ in groupby(data_id)])



















    
    
    
    








    
    
    
    
