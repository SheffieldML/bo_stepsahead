%%%%%%%%%%%%
% Javier Gonzalez
% 2015
% example to run the emin_epmgp
%%%%%%%%%%%%

% dimensionality of the problem
n = 10;

% make a Gaussian
m = zeros(n,1);
K = randn(n);
K = K*K' + eye(n);

% bounds for the minimum
eta = -10;

% calculate the integral with epmgp
[e_min,Int_y,Probs_y,Int_eta] = emin_epmgp(m,K,eta);

% sanity check
fprintf('--SANITY TESTS--\n')
fprintf('----------------\n')
fprintf('The sum of the probabilities in the polyhedra should be (apprx.) one: %g.\n',sum(Probs_y)+Int_eta);
fprintf('The expectation should be smaller than eta. This number should be positive: %g.\n',eta-e_min);
