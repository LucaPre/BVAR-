% Computes missing values using iterated PCA estimates following e Stock and Watson, 2002, Macroeconomic forecasting using diffusion
% indexes, J. Business and Economic Statistics 20, 147-162
function [datanew exitflag convergence] = FacMissing(dataraw,convval,maxfac,maxiter)
k=size(dataraw,2); % Number of variables
T=size(dataraw,1);
[dataraw,C,S]=normalize(dataraw);
datarawcopy=dataraw;
colsWithNaN = any(isnan(dataraw), 1);
dataraw(:,colsWithNaN) = [];
K=FacIC(dataraw,maxfac);
sigmax=cov(dataraw);
[lambda,sigma_f]=eigs(sigmax,K);
f=dataraw*lambda;
lambda=zeros(K,T);
datanew=zeros(T,k);
for i=1:k
y=datarawcopy(:,i);
rowsWithNaN = isnan(y);
y(rowsWithNaN) = [];
fcopy=f;
fcopy(rowsWithNaN, :) = [];
lambda(:,i)=(fcopy'*fcopy)^-1*fcopy'*y;
datanew(:,i)=datarawcopy(:,i);
yhat=f*lambda(:,i);
datanew(rowsWithNaN,i)=yhat(rowsWithNaN);
end

% Start iteration
convergence=1;
count=0;
exitflag=1;
while convergence>convval
    count=count+1;
datanewcopy=datanew;
sigmax=cov(datanew);
K=FacIC(datanew,maxfac);
[lambda,sigma_f]=eigs(sigmax,K);
f=datanew*lambda;
lambda=lambda';
for i=1:k
y=datarawcopy(:,i);
rowsWithNaN = isnan(y);
datanew(:,i)=datarawcopy(:,i);
yhat=f*lambda(:,i);
datanew(rowsWithNaN,i)=yhat(rowsWithNaN);
end
convergence=sum(sum((datanewcopy-datanew).^2));
if count==maxiter
    break
    exitflag=0;
end
end

datanew=datanew.*S;
datanew=datanew+C;


