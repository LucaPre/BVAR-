% Returns number of factors chosen by Bai & NG information criterion (PCA estimate)
function K = FacIC(X,maxfac)
IC=zeros(maxfac,1);
N=size(X,2);
T=size(X,1);
for i=1:length(IC)
    sigmax=cov(X);
[lambda,sigma_f]=eigs(sigmax,i);
f=X*lambda;
missfit=trace((X-f*lambda')'*(X-f*lambda'))/(N*T);
IC(i)=missfit+i*(N+T)/(N*T)*log(min(N,T));
end
K=find(IC==min(IC));