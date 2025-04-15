% Gets log-likelihood of GARCH(1,1) by recursive construction of variance
% and summation

function [Q, sigma_sq] = GARCHLikelihood(theta,u,sigma_initial)
T=length(u);

a0=theta(1);
a1=theta(2);
b1=theta(3);
LogL=zeros(T,1);
sigma_sq=zeros(T,1);
sigma_sq(1)=sigma_initial;
LogL(1)=-0.5*log(sigma_initial)-0.5*u(1)^2/sigma_initial;
for t=2:T
sigma_sq(t)=a0+a1*u(t-1)^2+b1*sigma_sq(t-1);
LogL(t)=-0.5*log(sigma_sq(t))-0.5*u(t)^2/sigma_sq(t);
end
Q=sum(LogL);



