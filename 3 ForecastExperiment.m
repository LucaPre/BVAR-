clear
clc
datatable = readtable('2024-05.csv');
dataarray=table2array(datatable(:,2:end));
transcode=dataarray(2,:);
transcode(116:130)=5; transcode(205:215)=5; transcode(233)=5; % Assume Price indices to be I(1) (in logs)
data=dataarray(3:end,:);

% Transform according to transformation code
datatransformed=zeros(size(data,1)-2,size(data,2));
for j=1:size(data,2)
    if transcode(j)==1
        datatransformed(:,j)=data(3:end,j);
    end
    if transcode(j)==2
        datatransformed(:,j)=data(3:end,j)-data(2:end-1,j);
    end
    if transcode(j)==3
        datatransformed(:,j)=(data(3:end,j)-data(2:end-1,j))-(data(2:end-1,j)-data(1:end-2,j));
    end
    if transcode(j)==4
        datatransformed(:,j)=log(data(3:end,j));
    end
    if transcode(j)==5
        datatransformed(:,j)=100*(log(data(3:end,j))-log(data(2:end-1,j)));
    end
    if transcode(j)==6
        datatransformed(:,j)=(log(data(3:end,j))-log(data(2:end-1,j)))-(log(data(2:end-1,j))-log(data(1:end-2,j)));
    end
    if transcode(j)==7
        datatransformed(:,j)=(data(3:end,j)./data(2:end-1,j)-1)-(data(2:end-1,j)./data(1:end-2,j)-1);
    end
end

% Replace missing data of panel
data=FacMissing(datatransformed,0.001,20,1000);

% Select variables to use
variablecode=([1 2 22 23 35 37 57 58 59 76 81 83 95 120 138 144 145 147 148 152 160 161 245]); 
Y=data(:,variablecode);
k=size(Y,2);

% Run Outlier Filter
Yfilt=Y;
breakstart=243;
breakend=245;
mulitpoints=4; % Number of starting points for numerical optimization (the more the longer computation takes but more likely to find the global optimum)
for j=1:k
    for t=1:breakend-breakstart+1
x0=[0 0 std(Y(:,j)) std(Y(:,j))*3];
opts = optimoptions(@fmincon,'Algorithm','interior-point','MaxFunctionEvaluations',1000,'Display','off');
ms = MultiStart('UseParallel',true,'Display','off','XTolerance',0.001,'FunctionTolerance',0.001);
fixedFunction = @(x) AR1FilterLik(x,Y(1:breakstart-1+t,j),breakstart); 
problem = createOptimProblem('fmincon','x0',x0,'objective',fixedFunction,'Aineq', [0 1 0 0; 0 0 -1 0; 0 0 0 -1; 0 -1 0 0], 'bineq', [1; 0; 0; 1],'options',opts);
[thetahat fval exitflag] = run(ms,problem,mulitpoints); 
[~,stateest]=AR1FilterLik(thetahat,Y(1:breakstart-1+t,j),breakstart);
Yfilt(breakstart-1+t,j)=stateest(breakstart-1+t);
    end
    j
end

dates=(1959.5:.25:2024)';
figure 
subplot(2,1,1)
plot(dates,Y(:,1))
title('Without Filter')
ylim([-9 9])
xlim([min(dates) max(dates)])
subplot(2,1,2)
plot(dates,Yfilt(:,1))
title('With Filter')
xlabel('Year')
ylim([-9 9])
xlim([min(dates) max(dates)])



%% Setup
Y=Yfilt;
h=8; % Forecast horizons
p=4; % Lags
S0=1000; % Burn-in
S1=10000; % Number of draws after burn-in
T_break=103;
nu0=[4 4]; % prior degree of freedom of lambda- 1, 2
ssq0=[0.04 0.04]; % prior mean of lambda1 and lambda2 (or inverse of the prior mean of the inverse of lambda for inverse gamma prior) 
ssq0lag=(1./(1:p)).^2; % prior mean of lambdalags
%ssq0lag=ones(1,4);
nu0lag=4; % Degree of freedom for lag shrinkage parameters
ssq0contemp=1/10000; % Prior mean of shrinkage factor for contemporaneous variables
nu0contemp=1000000; % Prior degree of freedom of contemporaneous shrinkage
IG=1; % indicator whether to use the inverse-gamma priors


%% Forecast experiment
parfevalOnAll(@() warning('off', 'MATLAB:nearlySingularMatrix'), 0);
rng(1)
seeds=randi([1 2^32-1],1000000,1);

% Out of Sample Forecast Statistics for Minnesota Type Priors (Takes very
% long as we need to perform MCMC 156 times, scales with number of
% available cores due to running in parallel)
[MSFE, ALPL, ALPLvec]=OutOfSampleStats(Y,h,T_break,p,S0,S1,nu0,ssq0,ssq0lag,nu0lag,ssq0contemp,nu0contemp,seeds,IG);

% Stats of AR1 Benchmark
[AR1MSE, AR1ALPL, ARALPLvec, Forecast]=AR1_MSFE(Y,h,T_break);


RMSEgains=zeros(h,k);
ALPLgains=zeros(h,k);
ALPLvecgains=zeros(h,1);
for j=1:h
outlier=[147:149]-j+1;
logicalIndex = true(156-j+1,1);
logicalIndex(outlier) = false;
RMSEgains(j,:)=(sqrt(mean(AR1MSE(logicalIndex,:,j)))./sqrt(mean(MSFE(logicalIndex,:,j)))-1)*100;
ALPLgains(j,:)=(mean(ALPL(logicalIndex,:,j))-mean(AR1ALPL(logicalIndex,:,j)))*100;
ALPLvecgains(j)=(mean(ALPLvec(logicalIndex,j))-mean(ARALPLvec(logicalIndex,j)))*100;
end


