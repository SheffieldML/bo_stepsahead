import GPyOptmsa
import GPyOpt
import GPy
from numpy.random import seed
import numpy as np
from GPyOptmsa.util.general import samples_multidimensional_uniform, multigrid

import warnings
warnings.filterwarnings("ignore")
seed(12345)

input_dim = 2

# Generate the function to optimize
k 		= GPy.kern.RBF(input_dim=input_dim,lengthscale=0.01) + GPy.kern.Bias(input_dim=input_dim)
bounds 	= [(0,1),(0,1)]
input_dim = len(bounds)

X1 = np.linspace(bounds[0][0], bounds[0][1], 25)
X2 = np.linspace(bounds[1][0], bounds[1][1], 25)
x1, x2 = np.meshgrid(X1, X2)
x = np.hstack((x1.reshape(25*25,1),x2.reshape(25*25,1)))

mu = np.zeros((25**2)) # vector of the means --- we could use a mean function here, but here it is just zero.
C = k.K(x,x) # compute the covariance matrix associated with inputs X

# Generate 20 separate samples paths from a Gaussian with mean mu and covariance C
seed(5)
z = np.random.multivariate_normal(mu,C,1).T

# point interpolation
m = GPy.models.GPRegression(x,z,kernel=k)
m.Gaussian_noise.constrain_fixed(1e-4)
m.optimize()


# objective fucntion

class objective_gp():
    def __init__(self,model):
        self.model = model
    def f(self,x):
        return self.model.predict(x)[0]

obj = objective_gp(m)  
f = obj.f


# --- Matrices to save results

NR = 10
n_init          = 5             # number of initial points (per dimension).
max_iter_dim    = 10            # Number of iterations (per dimension).


res_GLASSES_H    = np.empty([1,2])
res_EL            = np.empty([1,2])


# --- replicates 
for k in range(NR):
    
    print k
    seed(k)
    # --- inital points
    X = samples_multidimensional_uniform(bounds,n_init*len(bounds))
    Y = f(X)

    # --- Crete the optimization objects
    GLASSES_H    = GPyOptmsa.msa.GLASSES(f,bounds, X,Y,exact_feval=True)
    EL           = GPyOpt.methods.BayesianOptimization(f=f,bounds=bounds, X=X, Y=Y ,acquisition='EL',exact_feval=True) 


    print 'GLASSES_H'
    GLASSES_H.run_optimization(max_iter=max_iter_dim*input_dim,ahead_remaining = True)  
    rep_col     = [k]*GLASSES_H.Y_best.shape[0]
    res_GLASSES_H    = np.vstack((res_GLASSES_H,np.vstack((rep_col,GLASSES_H.Y_best)).T))
    np.savetxt('res_GLASSES_H_'+ '2D_2' +'.txt', res_GLASSES_H)

    print 'EL'
    EL.run_optimization(max_iter=max_iter_dim*input_dim, acqu_optimize_method='DIRECT')  
    rep_col     = [k]*EL.Y_best.shape[0]
    res_EL    = np.vstack((res_EL,np.vstack((rep_col,EL.Y_best)).T))
    np.savetxt('res_EL_'+ '2D_2' +'.txt', res_EL)


## resutls analysis

def value_at_last(data):
    data_val = data[:,1]
    data_id  = data[:,0]
    from itertools import groupby
    return np.array([data_val[data_id==i].min() for i, _ in groupby(data_id)])


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
results[0,:] = value_at_last(np.loadtxt('res_EL_'+'1D_2' +'.txt')[1:,:])
results[1,:] = value_at_last(np.loadtxt('res_GLASSES_H_'+'1D_2'+'.txt')[1:,:])
    
