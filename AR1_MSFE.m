function [MSFE, ALPL, ALPLvec, MSEForecast,actual] = AR1_MSFE(y,h,T_break)
k=size(y,2);
MSFE=zeros(length(y)-T_break,k,h);
MSEForecast=zeros(length(y)-T_break,k);
actual=zeros(length(y)-T_break,k);
ALPL=zeros(length(y)-T_break,k,h);
ALPLvec=zeros(length(y)-T_break,h);
for m=1:size(MSFE,1)
Yraw=y(1:T_break+m-h,:);
Traw=size(Yraw,1);
means=zeros(k,h);
vars=zeros(k,h);
for i=1:k
Y=Yraw(2:end,i);
X=[ones(length(Yraw(2:end,i)),1) Yraw(1:end-1,i)];
AR=(X'*X)^-1*X'*Y; % OLS/ML estimator of AR(1)
sigma=(Y-X*AR)'*(Y-X*AR)/Traw; % ML estimator of variance
Forecast=[1 Y(end)];
Forecast_error=zeros(1,h);
Forecast_final=zeros(1,h);
for j=1:h
Forecast=Forecast*AR;
Forecast_final(j)=Forecast; % h-step Forecast by AR(1) recursion
Forecast=[1 Forecast];
Forecast_error(j)=sum(AR(2).^(0:j-1))*sigma;
ALPL(m,i,j)=log(normpdf(y(Traw+j,i),Forecast_final(j),sqrt(Forecast_error(j))));
MSFE(m,i,j)=(Forecast_final(j)-y(Traw+j,i))^2;
if j==1
MSEForecast(m,i)=Forecast_final(j);
actual(m,i)=y(Traw+j,i);
end
means(i,j)=Forecast_final(j);
vars(i,j)=Forecast_error(j);
end
end
for j=1:h
ALPLvec(m,j)=-k/2*log(2*pi)-0.5*log(det(diag(vars(:,j))))-0.5*(y(Traw+j,:)-means(:,j)')*diag(vars(:,j))^-1*(y(Traw+j,:)-means(:,j)')';
end
end