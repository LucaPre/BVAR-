% Fits a (multivariate) t distribution by fitting mean and then iterating
% between fitting covariance and estimating degree of freedom by maximizing
% the likelihood

function [loglik, sigma, nu, my] = mvtfit(X,epsilon,x0)
p=size(X,2);

% Log-Likelihood
if p>1
logtdens = @(x,my,sigma,nu,p) log(gamma((nu+p)/2))-log(gamma(nu/2))-p/2*log(nu)-p/2*log(pi)-0.5*det(sigma)-(nu+p)/2.*log(1+1/nu*sum(((x-my)*sigma^-1.*(x-my))')');
else
logtdens = @(x,my,sigma,nu,p) log(gamma((nu+p)/2))-log(gamma(nu/2))-p/2*log(nu)-p/2*log(pi)-0.5*det(sigma)-(nu+p)/2.*log(1+1/nu*((x-my).^2*sigma^-1));
end

sigma=cov(X)*8/10;
nu=10;
my=mean(X);
conv=1e+50;
iter=0;
while conv>epsilon
    iter=iter+1;
fixedFunction=@(nu) -sum(logtdens(X,my,sigma,nu,p));
nuhat=fminbnd(fixedFunction,2,340-p);
conv=abs(nu-nuhat);
nu=nuhat;
sigma=(nuhat-2)/nuhat*cov(X);
if iter>100
    sigma=cov(X);
    nu=340-p;
    break
end
end

logtdens = @(x,my,sigma,nu,p) log(gamma((nu+p)/2))-log(gamma(nu/2))-p/2*log(nu)-p/2*log(pi)-0.5*log(det(sigma))-(nu+p)/2.*log(1+1/nu*((x-my)*sigma^-1*(x-my)'));
loglik=logtdens(x0,my,sigma,nu,p);
