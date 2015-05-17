#!/usr/bin/env python

"""
The script for comparing the performance of difference Bayesian Optimization methods.
"""

import GPyOpt
import time
import h5py
import numpy as np

# Configurations
model_data_init = 5                     # Number of random inital points  
max_iter = 40                           # Number of Batches
NR = 30                                 # Number of replicates of the experiment
acq_optimizer = 'fast_random'           # Acquisition optimizer
acqu_optimize_restarts = 200            # Random re-starts of the aquisition optimizer

result_file = 'cosines.h5'

methods_config = [
                  { 'name': 'GP-UCB',
                    'acquisition_name':'LCB',
                    'acquisition_par': 2,
                    'acqu_optimize_method':'fast_random',
                    'acqu_optimize_restarts':acqu_optimize_restarts,
                  }, 
                  { 'name': 'Expected-Loss',
                    'acquisition_name':'EL1',
                    'acquisition_par': 0,
                    'acqu_optimize_method':'fast_random',
                    'acqu_optimize_restarts':acqu_optimize_restarts,
                  },
                  { 'name': 'Expected-Loss-5',
                    'acquisition_name':'ELn',
                    'acquisition_par': 5,
                    'acqu_optimize_method':'fast_random',
                    'acqu_optimize_restarts':acqu_optimize_restarts,
                  },
                  { 'name': 'Expected-Loss-10',
                    'acquisition_name':'ELn',
                    'acquisition_par': 10,
                    'acqu_optimize_method':'fast_random',
                    'acqu_optimize_restarts':acqu_optimize_restarts,
                  },
                    { 'name': 'Expected-Loss-15',
                    'acquisition_name':'ELn',
                    'acquisition_par': 15,
                    'acqu_optimize_method':'fast_random',
                    'acqu_optimize_restarts':acqu_optimize_restarts,
                  },
                    { 'name': 'Expected-Loss-all',
                    'acquisition_name':'ELn',
                    'acquisition_par': 0,
                    'acqu_optimize_method':'fast_random',
                    'acqu_optimize_restarts':acqu_optimize_restarts,
                  },
    ]

def prepare_objective_function():
    f = GPyOpt.fmodels.experiments2d.cosines()
    return f.f, f.bounds


def get_random_initial_values(f,f_bounds,n_inits):
    from GPyOpt.util.general import samples_multidimensional_uniform
    xs = samples_multidimensional_uniform(f_bounds,n_inits)
    ys = f(xs)
    return xs, ys


if __name__ == '__main__':

    # Create the target function
    f_obj,f_bounds = prepare_objective_function()

    y_list = {}
    t_list = {}

    for i_exp in xrange(NR):

        # Get Initial evaluations
        xs_init, ys_init = get_random_initial_values(f_obj, f_bounds, model_data_init)

        for m_c in methods_config:

            print m_c['name']

            if m_c['name'] not in y_list:
                y_list[m_c['name']] = []
                t_list[m_c['name']] = []

            # Create the Bayesian Optimization Object
            bo = GPyOpt.methods.BayesianOptimization(f_obj, bounds=f_bounds, X= xs_init.copy(), Y=ys_init.copy(), acquisition=m_c['acquisition_name'], acquisition_par = m_c['acquisition_par'])

            
            t_list[m_c['name']].append([])
            y_list[m_c['name']].append([])

            start_time = time.time()
            for i_itr in xrange(max_iter):
                if i_itr==0:
                    rt = bo.run_optimization(max_iter = m_c['max_iter'], acqu_optimize_method=m_c['acqu_optimize_method'],acqu_optimize_restarts= m_c['acqu_optimize_restarts'],eps = -1, verbose=False)
                else:
                    rt = bo.run_optimization(max_iter = m_c['max_iter'], verbose=False)
                elapsed_time = time.time() - start_time
                t_list[m_c['name']][i_exp].append(elapsed_time)
                y_list[m_c['name']][i_exp].append(bo.Y_best[-1])

                if rt<1:
                    break


    # Write the results into a file
    f = h5py.File(result_file,'w')
    for m_c in methods_config:

        l_max = max([len(l) for l in t_list[m_c['name']]])
        ts = np.empty((len(t_list[m_c['name']]),l_max), dtype=np.float)
        ys = np.empty_like(ts)

        ts[:] = np.nan
        ys[:] = np.nan

        for i in xrange(len(t_list[m_c['name']])):
            t = t_list[m_c['name']][i]
            y = y_list[m_c['name']][i]
            ts[i,:len(t)] = t
            ys[i,:len(y)] = y

        # ts = np.array(t_list[m_c['name']])
        # ys = np.array(y_list[m_c['name']])

        d = f.create_dataset(m_c['name']+'_time', ts.shape, dtype=ts.dtype)
        d[:] = ts

        d = f.create_dataset(m_c['name']+'_Ybest', ys.shape, dtype=ys.dtype)
        d[:] = ys
    f.close()

