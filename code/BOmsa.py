import GPy
from GPyOpt.util.acquisitions import loss_nsahead


def BOmsa(f,bounds,X,Y,n_ahead,max_iter):

	# Create the GPyOpt object
	input_dim = len(bounds)
	kernel = GPy.kern.RBF(input_dim, variance=.1, lengthscale=.1)  + GPy.kern.Bias(input_dim)
	model  = GPy.models.GPRegression(X,Y,kernel=kernel)
	model.optimize()

	k = 1
	while k<=max_iter:

		
		# Evaluate the loss ahead acquisition function in a set of representer points
		x_acq = samples_multidimensional_uniform(bounds,500)
		y_acq = loss_nsahead(x_acq,n_ahead,model,bounds)

		# Build the acquisition: based on a model on the representer points
		kernel_acq      = GPy.kern.RBF(input_dim, variance=.1, lengthscale=.1)  + GPy.kern.Bias(input_dim)
		model_acq   	= GPy.models.GPRegression(x_acq,y_acq,kernel=kernel_acq)
		model_acq.optimize()

		# Optimize the posterior mean on the model and find the best location
		x_new = XXXXXX wrapper_lbfgsb(f,grad_f,x0,bounds)
		y_new = 

		# Augment the dataset
		Y = np.vstack((Y,Y_new))
		X = np.vstack((Y,Y_new))

		# Update the model
		model.X = X
		model.Y = Y
		model.optimize()


	



