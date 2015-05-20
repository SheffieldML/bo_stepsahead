from ..dpp_samplers.dpp import sample_dual_conditional_dpp
from ..quadrature.emin_epmgp import emin_epmgp
from ..util.general import samples_multidimensional_uniform, reshape
import numpy as np

def loss_nsahead(x, n_ahead, model, bounds):
    x = reshape(x,model.X.shape[1]) 
    n_data = x.shape[0]
    
    # --- fixed options
    num_data       = 500              # uniform samples
    n_replicates   = 5             # dpp replicates
    q              = 50             # truncation, for dual dpp

    # --- get values
    losses        = np.zeros((n_data,n_replicates))
    Y             = model.Y
    eta           = Y.min()
    
    X0            = samples_multidimensional_uniform(bounds,num_data)
    set1          = [1]
    
    if (n_ahead>1):
        for k in range(n_data):
            X             = np.vstack((x[k,:],X0))
       
            # --- define kernel matrix for the dpp
            L   = model.kern.K(X)
    
            for j in range(n_replicates):
                # --- take a sample from the dpp (need to re-index to start from zero)
                dpp_sample = sample_dual_conditional_dpp(L,set1,q,n_ahead)
                dpp_sample = np.ndarray.tolist(np.array(dpp_sample)-1)
          
                # evaluate GP at the sample and compute full covariance 
                m, K       = model.predict(X0[dpp_sample,:],full_cov=True)
       
                # compute the expected loss
                losses[k][j]  = emin_epmgp(m,K,eta)
    #else:
    #    loss =  fmin + (m-fmin)*Phi - s*phi  


    return losses.mean(1).reshape(n_data,1)


class acq_GPGOMSA:
    '''
    Wrapper class for a GPy model to minimize the posterion mean
    '''
    def __init__(self,model):
        self.model = model

    def f(self,x):
        return model.predict(x)[0]

    def d_f(self,x):
        dmdx = model.predictive_gradients(x)
        return dmdx[:,:,0]