trainnum=200;
r = sqrt(2*rand(trainnum,1)); % radius
t = 2*pi*rand(trainnum,1); % angle
data1 = [r.*cos(t), r.*sin(t)]; % points


r2 = sqrt(3*rand(trainnum,1)+1); % radius
t2 = 2*pi*rand(trainnum,1); % angle
data2 = [r2.*cos(t2), r2.*sin(t2)]; % points


figure;
plot(data1(:,1),data1(:,2),'r.')
hold on
plot(data2(:,1),data2(:,2),'b.')

ezpolar(@(x)1.4);ezpolar(@(x)1)
axis equal
hold off
% 
% 

subplot(1,2,1);
data3 = [data1;data2];
theclass = ones(trainnum*2,1);
theclass(1:trainnum) = -1;

cl = svmtrain(data3,theclass,'Kernel_Function','rbf',...
    'boxconstraint',10,'showplot',true,'rbf_sigma',0.1);
hold on
axis equal
%ezpolar(@(x)1.4);ezpolar(@(x)1)
hold off


subplot(1,2,2);
data3 = [data1;data2];
theclass = ones(trainnum*2,1);
theclass(1:trainnum) = -1;

cl = svmtrain(data3,theclass,'Kernel_Function','rbf',...
    'boxconstraint',10,'showplot',true,'rbf_sigma',1);
hold on
axis equal
%ezpolar(@(x)1.4);ezpolar(@(x)1)
hold off





% 
% data3 = [data1;data2];
% theclass = ones(200,1);
% theclass(1:100) = -1;
% 
% cl = svmtrain(data3,theclass,'Kernel_Function','rbf',...
%     'boxconstraint',Inf,'showplot',true);
% hold on
% axis equal
% ezpolar(@(x)1)
% hold off
% 
% 

% cl = svmtrain(data3,theclass,'Kernel_Function','polynomial',...
%     'boxconstraint',Inf,'showplot',true);
% hold on
% axis equal
% ezpolar(@(x)1)
% hold off
% 
% figure;
% SVMstruct = svmtrain(xtrain,ytrain,'Kernel_Function','linear');
% newClasses = svmclassify(SVMstruct,[ax1 ax2]);
% scatter(ax1,ax2,[],newClasses,'filled');
% hold on
% plot(x1(:,1),x1(:,2),'b+',x2(:,1),x2(:,2),'ro');
% hold on
% plot(x1(1:usedtrain,1),x1(1:usedtrain,2),'c+',x2(1:usedtrain,1),x2(1:usedtrain,2),'yo');




