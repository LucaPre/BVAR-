% Computes Log Likelihood of state space model with AR(1) state equation
% and yt=st+et as measurement equation with SD(et)=0 before breakdate and
% =sigma_c after
function [LogLik, sy] = AR1FilterLik(theta,y,breakdate)
alpha0=theta(1);
alpha1=theta(2);
sigma=theta(3);
sigma_c=theta(4);
T=size(y,1);
sy=zeros(T,1);
sy(1)=y(1);
LogLik=0;

% Likelihood before breakdate (observed states, simple AR(1))
for t=2:breakdate-1
    sy(t)=y(t);
    expectation=alpha1*y(t-1)+alpha0;
    variance=sigma^2;
    LogLik=LogLik-0.5*log(variance)-0.5*variance^-1*(y(t)-expectation)^2;
end

% Recursion for unobserved state
Vsy=(1/sigma^2+1/sigma_c^2)^-1;
Esy=Vsy*(y(t-1)/sigma_c^2+(alpha0+alpha1*sy(t-2))/sigma^2);
for t=breakdate:T
sy(t)=Esy;
Vsy1=(Vsy^-1-(alpha1^2*Vsy^-2/(Vsy^-1*alpha1^2+sigma^-2)))^-1;
Esy1=Vsy1*(Vsy^-1*alpha0-(alpha0*alpha1^2*Vsy^-2-Vsy^-1*sigma^-2*alpha1*Esy)/(Vsy^-1*alpha1^2+sigma^-2));
Vyy=(Vsy1^-1-Vsy1^-2/(Vsy1^-1+sigma_c^-2))^-1;
Eyy=Vyy*(Esy1*Vsy1^-1*sigma_c^-2/(Vsy1^-1+sigma_c^-2));
LogLik=LogLik-0.5*log(Vyy)-0.5*Vyy^-1*(y(t)-Eyy)^2;
Vsy=(1/sigma_c^2+Vsy1^-1)^-1;
Esy=Vsy*(1/sigma_c^2*y(t-1)+Vsy1^-1*Esy1);
end

LogLik=-LogLik;
