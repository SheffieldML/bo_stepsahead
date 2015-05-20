import numpy as np
import GPy
from GPyOpt.core.optimization import wrapper_lbfgsb, wrapper_DIRECT
from .util.general import samples_multidimensional_uniform, reshape
from .util.acquisition import loss_nsahead
import GPyOptmsa



class GPGOMSA:
    '''
    Class to run Bayesian Optimization with multiple steps ahead
    '''

    def __init__(self,f,bounds,X,Y):
        self.input_dim = len(bounds)
        self.bounds = bounds
        self.X = X
        self.Y = Y
        self.kernel = GPy.kern.RBF(self.input_dim, variance=.1, lengthscale=.1)  + GPy.kern.Bias(self.input_dim)
        self.model  = GPy.models.GPRegression(X,Y,kernel= self.kernel)
        self.model.optimize()

    def run_optimization(self,max_iter,n_ahead=None, eps= 10e-6):

        # Check the number of steps ahead to look at
        if n_ahead==None:
            self.n_ahead = max_iter
        else:
            self.n_ahead = n_ahead

        # initial stop conditions
        k = 1
        distance_lastX = np.sqrt(sum((self.X[self.X.shape[0]-1,:]-self.X[self.X.shape[0]-2,:])**2))

        while k<=max_iter and distance_lastX > eps:
            print k
            print n_ahead

            # Evaluate the loss ahead acquisition function in a set of representer points
            x_acq = samples_multidimensional_uniform(self.bounds,500)
            y_acq = loss_nsahead(x_acq,self.n_ahead,self.model,self.bounds)

            # Build the acquisition: based on a model on the representer points
            kernel_acq      = GPy.kern.RBF(self.input_dim, variance=.1, lengthscale=.1)  + GPy.kern.Bias(self.input_dim)
            model_acq       = GPy.models.GPRegression(x_acq,y_acq,kernel=kernel_acq)
            model_acq.optimize()

            #Create object to optimize
            def f(x):
            	x = np.atleast_2d(x)
                return model_acq.predict(x)[0]

            # Optimize the posterior mean on the model and find the best location
            samples = samples_multidimensional_uniform(self.bounds,500)
            x0 =  samples[np.argmin(f(samples))]
            X_new = wrapper_DIRECT(f,self.bounds)
            #x_new = wrapper_lbfgsb(f,d_f, x0, self.bounds)

            # Augment the dataset
            self.X = np.vstack((self.X,X_new))
            self.Y = np.vstack((self.Y,f(X_new)))

            # Update the model
            self.model.set_XY(self.X,self.Y)
            self.model.optimize()

            # Update steps ahead if needed
            if n_ahead==None:
                self.n_ahead -=1

            k += 1






