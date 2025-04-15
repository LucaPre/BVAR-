% Given data y and break T_break gives simulated MSFEs of Forecasting yt+h 
% with data up until T_break+i-1 for all i up until T-h

function [MSFE, ALPL, ALPLvec, MSEForecast] = OutOfSampleStats(y,h,T_break,p,S0,S1,nu0,ssq0,ssq0lag,nu0lag,ssq0contemp,nu0contemp,seeds,IG)
k=size(y,2);

% Place to store Statistics
MSFE=zeros(length(y)-T_break,k,h);
MSEForecast=zeros(length(y)-T_break,k);
ALPL=zeros(length(y)-T_break,k,h);
ALPLvec=zeros(length(y)-T_break,h);

% Loop over different samples for estimation (expanding window)
parfor m=1:size(MSFE,1)
rng(seeds(m))
Yraw=y(1:T_break+m-h,:); % Sample used for estimation 
[Traw, M] = size(Yraw);

problem=1;
     while problem==1
         try
if IG==1
% Estimation when inverse gamma priors are used
[Forecasts,lambda1vec,lambda2vec,lambdalagvec,nu0i,lambdaivec,garchparams,c] = IGSampler(Yraw,p,nu0,nu0lag,ssq0,ssq0lag,ssq0contemp,nu0contemp,S0,S1,h);   
else
% Estimation when gamma priors are used
[Forecasts,lambda1vec,lambda2vec,lambdalagvec,nu0i,lambdaivec,garchparams,c] = GammaSampler(Yraw,p,nu0,nu0lag,ssq0,ssq0lag,ssq0contemp,nu0contemp,S0,S1,h);   
end
problem=0;
         catch
             problem=1;
             disp("Problem")
             continue
         end
     end

MSEForecast(m,:,1)=mean(Forecasts(S0:end,:,1));
for j=1:h
MSFE(m,:,j)=(mean(Forecasts(S0:end,:,j))-y(Traw+j,:)).^2; % Error between point forecast and out of sample observation
for i=1:k
ALPL(m,i,j)=mvtfit(Forecasts(S0:end,i,j),0.01,y(Traw+j,i)); % ALPL by fitting t distribution

end

ALPLvec(m,j)=mvtfit(Forecasts(S0:end,:,j),0.01,y(Traw+j,:)); % VEC-ALPL

end
m
end