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
bounds 	= [(0,1.5),(0,1.5)]
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
    np.savetxt('res_MPI_'+ '2D_2' +'.txt', res_MPI)

    print 'EL'
    LCB.run_optimization(max_iter=max_iter_dim*input_dim, acqu_optimize_method='DIRECT')  
    rep_col     = [k]*LCB.Y_best.shape[0]
    res_LCB    = np.vstack((res_LCB,np.vstack((rep_col,LCB.Y_best)).T))
    np.savetxt('res_LCB_'+ '2D_2' +'.txt', res_LCB)
    
