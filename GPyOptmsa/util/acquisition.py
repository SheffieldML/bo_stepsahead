from ..dpp_samplers.dpp import sample_dual_conditional_dpp
from ..quadrature.emin_epmgp import emin_epmgp
from ..util.general import samples_multidimensional_uniform, reshape, get_moments, get_quantiles
import numpy as np

def loss_nsahead(x, n_ahead, model, bounds):
    x = reshape(x,model.X.shape[1]) 
    n_data = x.shape[0]

    # --- fixed options
    num_init_dpp    = 500              # uniform samples
    n_replicates    = 5                # dpp replicates
    q               = 50               # truncation, for dual dpp

    # --- get values
    losses_samples  = np.zeros((n_data,n_replicates))
    Y               = model.Y
    eta             = Y.min()
    
    X0              = samples_multidimensional_uniform(bounds,num_init_dpp)
    set1            = [1]
    
    if n_ahead>1:                  
        # --- We need to this separatelly for each data points
        for k in range(n_data):
            X             = np.vstack((x[k,:],X0))
       
            # --- define kernel matrix for the dpp
            L   = model.kern.K(X)
    
            # --- averages of the dpp samples
            for j in range(n_replicates):
                # --- take a sample from the dpp (need to re-index to start from zero)
                dpp_sample = sample_dual_conditional_dpp(L,set1,q,n_ahead)
                dpp_sample = np.ndarray.tolist(np.array(dpp_sample)-1)
          
                # evaluate GP at the sample and compute full covariance 
                m, K       = model.predict(X0[dpp_sample,:],full_cov=True)
       
                # compute the expected loss
                losses_samples[k][j]  = emin_epmgp(m,K,eta)

        losses = losses_samples.mean(1).reshape(n_data,1)
    
    elif n_ahead ==1:               
        m, s, fmin = get_moments(model, x)
        phi, Phi, _ = get_quantiles(0, fmin, m, s)                 # self.acquisition_par should be zero
        losses =  fmin + (m-fmin)*Phi - s*phi                      # same as EI excepting the first term fmin

    return losses

