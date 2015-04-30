% problem bounds
lB = [-pi/2,-pi/2];
uB = [pi/2,pi/2];

% plot true function
[X1_grid,X2_grid] = meshgrid(lB(1):uB(1)/30:+uB(1),lB(2):uB(2)/30:+uB(2));
x_grid = [X1_grid(:) X2_grid(:)];
y_grid = sin(X1_grid).*sin(X2_grid);
imagesc(y_grid); drawnow;

% sample points 
n  = 10;
X1 = unifrnd(lB(1),uB(1),n,1);
X2 = unifrnd(lB(2),uB(2),n,1);
Y  = sin(X1).*sin(X2) + 0.17*randn(size(X1)) ;
x = [X1(:) X2(:)];
y = Y(:);

%% GP covariance
covfunc = @covSEiso; 
likfunc = @likGauss; 

hyp2.cov = [0 ; 0];    
hyp2.lik = log(0.1);
hyp2 = minimize(hyp2, @gp, -100, @infExact, [], covfunc, likfunc, x, y);

% training
nlml2 = gp(hyp2, @infExact, [], covfunc, likfunc, x, y);

% prediction
[m, S] = gp(hyp2, @infExact, [], covfunc, likfunc, x, y, x_grid);
m = reshape(m, size(y_grid));
figure(2); imagesc(m); drawnow;

%%
GP          = struct;
GP.hyp2     = hyp2;
GP.inf      = {@infExact};
GP.kernel   = covfunc;
GP.likfunc  = likfunc;
GP.x        = x;
GP.y        = y;

eta    = min(y);
x_star = [pi/2,pi/2];


%% compute the value of the loss with different look-ahead steps
n_ahead = 5;
n_grid  = max(size(x_grid));
loss10  = zeros(1,n_grid);

for j=1:n_grid
    j
    [loss,v_loss] = loss_msahead(x_grid(j,:), n_ahead, lB, uB, GP);
    loss10(j) = loss;
end

figure(3); imagesc(reshape(loss10,61,61)); drawnow;

