import GPyOptmsa
import GPyOpt
import GPy
from numpy.random import seed
import numpy as np
from GPyOptmsa.util.general import samples_multidimensional_uniform, multigrid

import warnings
warnings.filterwarnings("ignore")
seed(12345)

# Generate the function to optimize
k = GPy.kern.RBF(input_dim=1,lengthscale=0.01) + GPy.kern.Bias(input_dim=1)

x = np.linspace(0.,1.,500) # define X to be 500 points evenly spaced over [0,1]
x = x[:,None] # reshape X to make it n*p --- we try to use 'design matrices' in GPy 

mu = np.zeros((500)) # vector of the means --- we could use a mean function here, but here it is just zero.
C = k.K(x,x) # compute the covariance matrix associated with inputs X

# Generate 20 separate samples paths from a Gaussian with mean mu and covariance C
seed(5)
z = np.random.multivariate_normal(mu,C,1).T

# point interpolation
m = GPy.models.GPRegression(x,z,kernel=k)
m.optimize()


# objective fucntion

class objective_gp():
    def __init__(self,model):
        self.model = model
    
    def f(self,x):
        return self.model.predict(x)[0]

obj = objective_gp(m)  
f = obj.f
bounds = [(0,1)]


# --- Matrices to save results

NR = 10
n_init          = 5             # number of initial points (per dimension).
max_iter_dim    = 10            # Number of iterations (per dimension).
input_dim = len(bounds)

res_GLASSES_H    = np.empty([1,2])
res_EL            = np.empty([1,2])
res_MPI = np.empty([1,2])
res_LCB = np.empty([1,2])

# --- replicates 
for k in range(NR):
    
    print k
    seed(k)
    # --- inital points
    X = samples_multidimensional_uniform(bounds,n_init*len(bounds))
    Y = f(X)

    # --- Crete the optimization objects
    #GLASSES_H    = GPyOptmsa.msa.GLASSES(f,bounds, X,Y,exact_feval=True)
    #EL           = GPyOpt.methods.BayesianOptimization(f=f,bounds=bounds, X=X, Y=Y ,acquisition='EL',exact_feval=True) 
    MPI          = GPyOpt.methods.BayesianOptimization(f=f,bounds=bounds, X=X, Y=Y ,acquisition='MPI',exact_feval=True) 
    LCB           = GPyOpt.methods.BayesianOptimization(f=f,bounds=bounds, X=X, Y=Y ,acquisition='LCB',acquisition_par=1,exact_feval=True) 
   

    print 'MPI'
    MPI.run_optimization(max_iter=max_iter_dim*input_dim,acqu_optimize_method='DIRECT')  
    rep_col     = [k]*MPI.Y_best.shape[0]
    res_MPI    = np.vstack((res_MPI,np.vstack((rep_col,MPI.Y_best)).T))
    np.savetxt('res_MPI_'+ '1D_2' +'.txt', res_MPI)

    print 'EL'
    LCB.run_optimization(max_iter=max_iter_dim*input_dim, acqu_optimize_method='DIRECT')  
    rep_col     = [k]*LCB.Y_best.shape[0]
    res_LCB    = np.vstack((res_LCB,np.vstack((rep_col,LCB.Y_best)).T))
    np.savetxt('res_LCB_'+ '1D_2' +'.txt', res_LCB)


n_exp = 1
methods = ['EL','GL-H']

results_mean = np.zeros((n_exp,len(methods)))
results_min = np.zeros((n_exp,len(methods)))
results_median = np.zeros((n_exp,len(methods)))


for k in range(len(experiments)): 
    results_mean[k,:] = create_table(experiments[k]).mean(1)
    results_min[k,:] = create_table(experiments[k]).min(1)
    results_median[k,:] = np.median(create_table(experiments[k]),1)

 
results = np.zeros((2,5))
results[0,:] = value_at_last(np.loadtxt('res_MPI_'+'1D_2' +'.txt')[1:,:])
results[1,:] = value_at_last(np.loadtxt('res_LCB_'+'1D_2'+'.txt')[1:,:])
    
def value_at_last(data):
    data_val = data[:,1]
    data_id  = data[:,0]
    from itertools import groupby
    return np.array([data_val[data_id==i].min() for i, _ in groupby(data_id)])
