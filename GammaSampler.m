% Code mostly similar to IGSampler so I refer to that for comments. Only
% difference is that gamma priors are used for lambda1, lambda2, lambdalag
% and therefore posterior draws there are taken from the generalized 
% inverse gaussian distribution

function [Forecasts,lambda1vec,lambda2vec,lambdalagvec,nu0i,lambdaivec,garchparams,c] = GammaSampler(Yraw,p,nu0,nu0lag,ssq0,ssq0lag,ssq0contemp,nu0contemp,S0,S1,h)

[Traw, M] = size(Yraw);
X = [ones(Traw-p,1) getLags(Yraw,p)];
K = width(X);
Y = Yraw(p+1:end,:);
A_prior = zeros(M*p+1,M); 
a_prior = A_prior(:);    

% Get constant variances from fitted AR(1)s for scaling 
sigma_sq = zeros(M,1);
    for i = 1:M
        Ylag_i = [ones(Traw-1,1) getLags(Yraw(:,i),1)];
        Y_i = Yraw(2:end,i);
        alpha_i = (Ylag_i'*Ylag_i)^-1*Ylag_i'*Y_i;
        sigma_sq(i) = (Y_i-Ylag_i*alpha_i)'*(Y_i-Ylag_i*alpha_i)/(Traw-1);
    end 

   lambda_inter=10000;
   lambda_cont=ssq0contemp;
   intind=zeros(K,M); 
   intindscale=zeros(K,M);
   otherind=zeros(K,M); 
   otherindscale1=zeros(K,M); 
   otherindscale2=zeros(K,M); 
   lagindicator=zeros(K,M);

   ind = zeros(M,p);
   for i=1:M
        ind(i,:) = 1+i:M:K;
    end
    for i = 1:M  
        for j = 1:K   
                if j==1 
                    intind(j,i) = 1;   
                    intindscale(j,i) = 1/sigma_sq(i,1);
                elseif find(j==ind(i,:))>0
                    lagindicator(j,i)=ceil((j-1)/M);
                else
                    for kj=1:M
                        if find(j==ind(kj,:))>0
                            ll = kj;                   
                        end
                    end                  
                    lagindicator(j,i)=ceil((j-1)/M);
                    otherind(j,i)=1;
                    otherindscale1(j,i)=1/sigma_sq(i,1);
                    otherindscale2(j,i)=sigma_sq(ll,1);
                end
        end
    end
otherindvec=otherind(:); 
otherindscale1vec=otherindscale1(:); 
otherindscale2vec=otherindscale2(:);
intindvec=intind(:);
intindscalevec=intindscale(:); 
lagindicatorvec=lagindicator(:); 
A0_scale=ones(M,M); A0_scale=triu(A0_scale); 
for i=1:M
A0_scale(i,:)=A0_scale(i,:)./sigma_sq(i);
A0_scale(:,i)=A0_scale(:,i)*sigma_sq(i);
end     

S=S0+S1;
lambda1vec=zeros(S,1);
lambda2vec=zeros(S,1);
lambdacontvec=zeros(S,1);
lambdalagvec=zeros(S,p);
lambdaivec=zeros(S,length(a_prior));
a_draws=zeros(S,length(a_prior));
a_draws(1,:)=a_prior'+0.00001*sqrt(otherindscale2(:)')+0.00001*(1-otherindvec)';
lambda1vec(1)=ssq0(1);
lambdalagvec(1,:)=ssq0lag;
lambdaivec(1,:)=1;
nu01=nu0(1);
nu02=nu0(2);
nu0i=zeros(S,1); 
nu0i(1)=2;
garchparams=zeros(M,3,S)+0.001;
garchparams(:,1,1)=sigma_sq;
A_draw=zeros(K,M);
A0_draw=eye(M);
Forecasts=zeros(S,M,h);
accrates=zeros(S,M+1);
c=ones(S,M+1);
     
for s=2:S

a_draw=a_draws(s-1,:)';
lambda1=lambda1vec(s-1);
lambdalag=lambdalagvec(s-1,:);
lambdai=lambdaivec(s-1,:);

lagind = zeros(M*p+1, M);

for l = 1:p
    lagind(M*(l-1)+2:M*l+1, :) = 1/lambdalag(l); % Inverse of lag specific shrinkage
end

%% Draw lambda2
V_i_inv=diag(1/lambda1*otherind(:).*lagind(:).*otherindscale2(:).*otherindscale1(:)./lambdai'); % Inverse of prior Variance without lambda2 (and zeros on all elements not associated with lambda2) 
p1=(-p*(M-1)*M+nu02)/2;
a1=nu02/ssq0(2);
b1=(a_draw-a_prior)'*V_i_inv*(a_draw-a_prior);
lambda2=gigrnd(p1,a1,b1,1);
lambda2vec(s)=lambda2;

%% Draw lambdai
lagindvec=lagind(:);
for l=1:length(a_prior)
p1=(-1+nu0i(s-1))/2;
a1=nu0i(s-1);
V_i_inv=1/lambda1*(1/lambda_inter*intindscalevec(l)*intindvec(l)+(1-intindvec(l))*otherindvec(l)*lagindvec(l)*otherindscale2vec(l)*otherindscale1vec(l)/lambda2+(1-intindvec(l))*(1-otherindvec(l))*1*lagindvec(l)); % Inverse of prior without lambdai
b1=(a_draw(l)-a_prior(l))^2*V_i_inv;
lambdaivec(s,l)=gigrnd(p1,a1,b1,1);
end
lambdai=lambdaivec(s,:)';


%% Draw lambda1
V_i_inv=diag(1./lambdai.*((1-intindvec).*otherindvec.*lagindvec.*otherindscale2vec.*otherindscale1vec/lambda2+(1-intindvec).*(1-otherindvec).*lagindvec));
p1=(-(length(a_prior)-M)+nu01)/2;
a1=nu01/ssq0(1);
b1=(a_draw-a_prior)'*V_i_inv*(a_draw-a_prior);
lambda1=gigrnd(p1,a1,b1,1);
lambda1vec(s)=lambda1;

%% Draw lambdalag
for l=1:p
if l==1
    lambdalagvec(s,l)=1;
else
V_i_inv=(lagindicatorvec==l).*1./lambdai.*(otherindvec.*otherindscale2vec.*otherindscale1vec/lambda2+(1-otherindvec))/lambda1;
p1=(-M^2+nu0lag)/2;
a1=nu0lag/ssq0lag(l);
b1=(a_draw-a_prior)'*diag(V_i_inv)*(a_draw-a_prior);
lambdalagvec(s,l)=gigrnd(p1,a1,b1,1);
end
lambdalag(l)=lambdalagvec(s,l);
end
lagind = zeros(M*p+1, M);
for l = 1:p
    lagind(M*(l-1)+2:M*l+1,:) = 1/lambdalag(l);
end
lagindvec=lagind(:);

%% Draw volatilities and GARCH parameters
sigma_garch=zeros(size(Y,1),M);
for j=1:M
Xtilde=Y;
Xtilde(:,1:j)=[];
u=Y(:,j)-[X(:,:) Xtilde(:,:)]*[A_draw(:,j);A0_draw(j,j+1:end)'];

thetaold=garchparams(j,:,s-1);

if s<100
    propscale=[0.4482    0.0145   -0.4244
    0.0145    0.3948    0.1107
   -0.4244    0.1107    0.6669];
    propvar=propscale;
else
    propscale=cov(log(squeeze(garchparams(j,:,1:s-1))'));
    c(s,j+1)=min(100,exp(log(c(s-1,j+1))+1/(s-99)^0.4*(mean(accrates(1:s-1,j+1))-0.25)));
    propvar=c(s,j+1)*propscale+0.0001*eye(3);
end
thetastar=exp(log(thetaold)+mvnrnd(zeros(1,3),propvar));
if sum(thetastar(2:3))>1
    acc=0;
else
acc=exp(GARCHLikelihood(thetastar,u,sigma_sq(j))+sum(log(thetastar))-GARCHLikelihood(thetaold,u,sigma_sq(j))-sum(log(thetaold))); % due to diffuse prior and change of variables posterior of log(theta) is proportional to likelihood times product of exp(log(theta)) (determinant of jacobian)
end

if rand(1)<min(acc,1)
    garchparams(j,:,s)=thetastar;
    accrates(s,j+1)=1;
else
    garchparams(j,:,s)=thetaold;
end
[~, sigmasfit] = GARCHLikelihood(garchparams(j,:,s),u,sigma_sq(j));
sigma_garch(:,j)=sigmasfit;
end

%% Draw parameters
V_prior_inv=(1./lambdai.*(1/lambda_inter.*intindscalevec.*intindvec*lambda1.*lambdai+(1-intindvec).*otherindvec.*lagindvec.*otherindscale2vec.*otherindscale1vec/lambda2+(1-intindvec).*(1-otherindvec).*lagindvec))/lambda1;

A_draw=zeros(K,M);
A0_draw=eye(M);
U=zeros(M,1);

% Equation by equation
for j=1:M
Xtilde=Y;
Xtilde(:,1:j)=[];
V_i_prior_inv=diag([V_prior_inv(1+(j-1)*K:j*K,:) ; 1/lambda_cont/sigma_sq(j)*sigma_sq(j+1:end)]);
L = chol([X Xtilde]'*diag(1./sigma_garch(:,j))*[X Xtilde] + V_i_prior_inv, 'lower');
fac = L \ ([X Xtilde]'*diag(1./sigma_garch(:,j))*Y(:,j));
a_i_post = L' \ fac;
z = randn(size(a_i_post));
a_i_draw = a_i_post + L' \ z;
A_draw(:,j)=a_i_draw(1:K);
A0_draw(j,j)=1;
A0_draw(j,j+1:end)=a_i_draw(K+1:end)';
U(j)=Y(end,j)-[X(end,:) Xtilde(end,:)]*a_i_draw;
end
a_draw=A_draw(:);
a_draws(s,:)=a_draw';

%% Draw Forecast
X_Forecast=[Y(end,:) X(end,2:end-M)];

sigma_forecast=sigma_garch(end,:);
for j=1:h
    Forecast=zeros(1,M);
    for i=flip(1:M)
        sigma_forecast(i)=garchparams(i,:,s)*[1; U(i)^2; sigma_forecast(i)];
        Xtilde=Forecast;
        Xtilde(:,1:i)=[];
        Forecast(i)=[1 X_Forecast Xtilde]*[A_draw(:,i);A0_draw(i,i+1:end)']+sqrt(sigma_forecast(i))*randn(1);
        U(i)=Forecast(i)-[1 X_Forecast Xtilde]*[A_draw(:,i);A0_draw(i,i+1:end)'];
    end
Forecasts(s,:,j)=Forecast;
X_Forecast=[Forecast X_Forecast(1:end-M)];
end

%% Draw degree of freedom of local shrinkage factor
if s<100
    propscale=0.0369;
    propvar=propscale;
else
    propscale=var(log(nu0i(1:s-1)));
    c(s,1)=min(100,exp(log(c(s-1,1))+1/(s-99)^0.4*(mean(accrates(1:s-1,1))-0.25)));
    propvar=c(s,1)*propscale;
end

nu0istar=exp(log(nu0i(s-1))+randn(1)*sqrt(propvar));
logpost=@(nu) nu/2*(length(a_prior)-M)*log(nu/2)-(length(a_prior)-M)*log(gamma(nu/2))-0.5*nu*(1+sum((1-intindvec).*lambdai))+nu/2*sum((1-intindvec).*log(lambdai));
acc=min(1,exp(logpost(nu0istar)+log(nu0istar)-log(nu0i(s-1))-logpost(nu0i(s-1))));
accrates(s,1)=acc;
if rand(1)<acc
    nu0i(s)=nu0istar;
else
    nu0i(s)=nu0i(s-1);
end

%% Draw shrinkage of contemporaneous coefficients
nu1=(M^2-M)/2+nu0contemp;
V_i_inv=diag(A0_scale(triu(true(size(A0_scale)),1)));
ssq1=((A0_draw(triu(true(size(A0_draw)),1)))'*V_i_inv*(A0_draw(triu(true(size(A0_draw)),1)))+nu0contemp*ssq0contemp)/nu1;
lambda_cont=1/gamrnd(nu1/2,(ssq1*nu1/2)^-1,1);
lambdacontvec(s)=lambda_cont;

if mod(s,100)==0
    s
end

end