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

data=FacMissing(datatransformed,0.001,20,1000);
dates=(1959.5:.25:2024)';
% Select variables to use
variablecode=[1 2 22 23 35 37 57 58 59 76 81 83 95 120 138 144 145 147 148 152 160 161 245]; 
Y=data(:,variablecode);
k=size(Y,2);

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
Y=Yfilt;


%% Setup
h=8; % Forecast horizons
p=4; % Lags
S0=1000; % Burn-in
S1=10000; % Number of draws after burn-in
nu0=[4 4]; % prior degree of freedom of lambda- 1, 2
ssq0=[0.04 0.04]; % prior mean of lambda1 and lambda2 (or inverse of the prior mean of the inverse of lambda for inverse gamma prior) 
ssq0lag=(1./(1:p)).^2; % prior mean of lambdalags
%ssq0lag=ones(1,4);
nu0lag=4;
ssq0contemp=1/10000;
nu0contemp=1000000;

%% Estimation with full sample (Drawing 6000 times takes 687 second on my laptop)
rng(1)

[Forecasts,lambda1vec,lambda2vec,lambdalagvec,nu0i,lambdaivec,garchparams,c,accrates] = IGSampler(Y,p,nu0,nu0lag,ssq0,ssq0lag,ssq0contemp,nu0contemp,S0,S1,h);


%% Summary of full sample results
mean(lambda1vec(S0:end))
mean(lambda2vec(S0:end))
mean(lambdalagvec(S0:end,:))
mean(nu0i(S0:end))
min(mean(lambdaivec(S0:end,:)))
max(mean(lambdaivec(S0:end,:)))
mean(accrates)

std(lambda1vec(S0:S0+S1))
std(lambda2vec(S0:S0+S1))
std(lambdalagvec(S0:S0+S1,:))
std(nu0i(S0:S0+S1))

% Convergence tests of Forecasts
tests=zeros(k,h);
for i=1:k
    for j=1:h
        A=0.1*S1; B=0.5*S1;
        CD=(mean(Forecasts(S0:S0+A,i,j))-mean(Forecasts(S0+A+B:S0+S1,i,j)))/sqrt(NeweyWest(Forecasts(S0:S0+A,i,j))/A+NeweyWest(Forecasts(S0+A+B:S0+S1,i,j))/(S1-A-B));
        tests(i,j)=abs(CD)>1.96;
    end
end
mean(mean(tests))

%% Trace Plots

% Lambda1
figure
plot(lambda1vec(S0:end),'DisplayName', '\lambda_1')
xlim([1 S1+1])
legend('Location','best','FontSize',15)


% Lambda2
figure
plot(lambda2vec(S0:end),'DisplayName', '\lambda_2')
xlim([1 S1+1])
legend('Location','best','FontSize',15)

% Lambdalags
figure
for i=2:p
subplot(1,3,i-1)
name=string(i);
plot(lambdalagvec(S0:end,i),'DisplayName', ['$\lambda_{l=' num2str(i) '}$'])
xlim([1 S1+1])
legend('Location','north','FontSize',12,'Interpreter','Latex')
end

% nu0i
figure
plot(nu0i(S0:end),'DisplayName', '\nu_k')
xlim([1 S1+1])
legend('Location','best','FontSize',15)

% Forecasts (GDP only)
figure
for i=1:h
    subplot(2,4,i)
    plot(Forecasts(S0:end,1,i),'DisplayName',['h=' num2str(i)])
    xlim([1 S1+1])
    title(['h=' num2str(i)])
end

% garch parameters 
names={'$a_0$','$a_1$','$b_1$'};
figure
for i=1:3
subplot(1,3,i)
plot(squeeze(garchparams(1,i,S0:end)),'DisplayName', names{i})
xlim([1 S1+1])
legend('Location','north','FontSize',12,'Interpreter','Latex')
end


nuhats=zeros(h,k);
for i=1:h
    for j=1:k
[~,~,nuhat,~]=mvtfit(Forecasts(S0:end,j,i),0.0001,Forecasts(end,j,i));
nuhats(i,j)=nuhat;
    end
end

nuhats=zeros(h,1);
for i=1:h
[~,~,nuhat,~]=mvtfit(Forecasts(S0:end,:,i),0.0001,Forecasts(end,:,i));
nuhats(i)=nuhat;
end
