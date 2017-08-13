trainnum=100;
pi=3.1415926;
xs=-2*pi;
xe=2*pi;
noise_m=0;
noise_d=1;
mx=xs:0.1:xe;
sinx=sin(mx);

x = xs + (xe-xs).*rand(trainnum,1);
y=x.*x.*x+ noise_m + noise_d.*randn(trainnum,1);
data1=[x y];


x = xs + (xe-xs).*rand(trainnum,1);
y=100+x.*x.*x+ noise_m + noise_d.*randn(trainnum,1);
data2=[x y];


% subplot(1,2,1);
data3 = [data1;data2];
theclass = ones(trainnum*2,1);
theclass(1:trainnum) = -1;

cl = svmtrain(data3,theclass,'Kernel_Function','rbf',...
    'boxconstraint',100,'showplot',true,'rbf_sigma',1);
hold on
%ezpolar(@(x)1.4);ezpolar(@(x)1)
hold off

figure
data3 = [data1;data2];
theclass = ones(trainnum*2,1);
theclass(1:trainnum) = -1;

cl = svmtrain(data3,theclass,'Kernel_Function','rbf',...
    'boxconstraint',1,'showplot',true,'rbf_sigma',1);
hold on
%ezpolar(@(x)1.4);ezpolar(@(x)1)
hold off
% 
figure
data3 = [data1;data2];
theclass = ones(trainnum*2,1);
theclass(1:trainnum) = -1;

cl = svmtrain(data3,theclass,'Kernel_Function','polynomial',...
    'boxconstraint',100,'showplot',true,'polyorder',4);
hold on

%ezpolar(@(x)1.4);ezpolar(@(x)1)
hold off


