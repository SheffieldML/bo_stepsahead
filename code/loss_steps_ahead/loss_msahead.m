%%%%%%%%%%%%
% Javier Gonzalez
% 2015
% code to computer the expected loss multiple steps ahead
% -- Input 
% x     : putative point
% eta   : current minimum
% n     : number of steps ahead
% lB    : lower bounds 
% uB    : upper bounds
% kernel: kerel function
% -- Output
% loss  : computed expected loss
% v_loss: vector of comuted losses for each dpp sample
%%%%%%%%%%%%
function [loss,v_loss] = loss_msahead(x_star, n_ahead, lB, uB, GP)
    % --- fixed options
    n_bigset_dpp = 500;
    n_replicates = 10;
    q            = 50;

    % --- get constants 
    p           = length(x_star);
    v_loss      = zeros(1,n_replicates);
    x           = GP.x;
    y           = GP.y;
    kernel      = GP.kernel;
    hyp2        = GP.hyp2;
    inf         = GP.inf;
    likfunc     = GP.likfunc;
    eta         = min(y);
    
    % --- generate candidate points (uniformly) and attach x
    X0  = unifrnd(repmat(lB,n_bigset_dpp,1),repmat(uB,n_bigset_dpp,1));
    X   = [x_star;X0];
    set = [1];
    
    % --- define kernel matrix for the dpp
    L   = kernel(hyp2.cov,X,X);
    
    for j = 1:n_replicates
        % --- take a sample from the dpp
        dpp_sample = sample_dual_conditional_dpp(L,q,set,n_ahead);
        
        % evaluate GP at the sample and compute full covariance 
        xTarget     = X(dpp_sample,:);
        [m, ~]      = gp(hyp2, inf, [], kernel, likfunc, x, y, xTarget);
        kTT         = kernel(hyp2.cov,xTarget,xTarget);
        kTI         = kernel(hyp2.cov,xTarget,x);
        kII         = kernel(hyp2.cov,x, x);
        sigma       = (exp(hyp2.lik)^2)*eye(max(size(x))); 
        Kpost       = kTT - kTI*inv(kII+sigma)*kTI';
        
        % compute thee expected loss
        [e_min,~,~,~] = emin_epmgp(m,Kpost,eta);
        v_loss(j) = e_min;
    end
    loss = mean(v_loss);
end
