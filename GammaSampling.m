% Performs sampling from a gamma distribution with Metropolis Hastings to
% compare sampling from the level to sampling from the log in accuracy of
% estimating first two moments. 

rng(1)
MC=1000; 
ssq=2; 
nu=6;
truevariance=2*ssq^-2/nu;
X=gamrnd(nu/2,(ssq*nu/2)^-1,10000,1);
guess1=var(X);
guess2=var(log(X));
loggamkern=@(x) (nu-2)/2*log(x)-0.5*x*nu*ssq; 
S0=1000;
S1=5000;
S=S0+S1;
dist=zeros(MC,4);
for m=1:MC
rawsample=ones(S,1);
logsample=ones(S,1);
accrates=zeros(S,2);
c=ones(S,2);
propvars=zeros(S,2);
for s=2:S
if s<100
    rawpropvar=guess1;
    logpropvar=guess2;
else
    c(s,1)=min(100,exp(log(c(s-1,1))+1/(s-99)^0.4*(mean(accrates(1:s-1,1))-0.25)));
    c(s,2)=min(100,exp(log(c(s-1,2))+1/(s-99)^0.4*(mean(accrates(1:s-1,2))-0.25)));

    rawpropvar=c(s,1)*var(rawsample(1:s-1));
    logpropvar=c(s,2)*var(log(logsample(1:s-1)));
    propvars(s,:)=[rawpropvar logpropvar];
end


rawprop=rawsample(s-1)+sqrt(rawpropvar)*randn(1); % Proposal from level sample
if rawprop<0
    acc=0;
else
    acc=exp(loggamkern(rawprop)-loggamkern(rawsample(s-1)));
end
accrates(s,1)=min(acc,1);
if rand(1)<min(acc,1)
    rawsample(s)=rawprop;
else 
    rawsample(s)=rawsample(s-1);
end

logprop=exp(log(logsample(s-1))+sqrt(logpropvar)*randn(1)); % Proposal from log sampling
acc=exp(loggamkern(logprop)+log(logprop)-loggamkern(logsample(s-1))-log(logsample(s-1)));
accrates(s,2)=min(acc,1);
if rand(1)<min(acc,1)
    logsample(s)=logprop;
else 
    logsample(s)=logsample(s-1);
end

end
% MSE between true mean and means from level and log sampling
dist1=100*([mean(rawsample(S0:end)) mean(logsample(S0:end))]-1/ssq).^2;

% IS distance between true variance and variance from level and log
% sampling
dist2=100*[var(rawsample(S0:end))/truevariance-1-log(var(rawsample(S0:end))/truevariance) var(logsample(S0:end))/truevariance-1-log(var(logsample(S0:end))/truevariance)];
dist(m,1:2)=dist1;
dist(m,3:4)=dist2;
m

% 2nd and 4th column show average distances of log sampling, 1st and 3rd
% from level sampling
mean(dist(1:m,:))
end

