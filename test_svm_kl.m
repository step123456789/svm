
r = sqrt(rand(100,1)); % radius
t = 2*pi*rand(100,1); % angle
data1 = [r.*cos(t), r.*sin(t)]; % points


r2 = sqrt(3*rand(100,1)+1); % radius
t2 = 2*pi*rand(100,1); % angle
data2 = [r2.*cos(t2), r2.*sin(t2)]; % points



plot(data1(:,1),data1(:,2),'r.')
hold on
plot(data2(:,1),data2(:,2),'b.')
ezpolar(@(x)1);ezpolar(@(x)2);
axis equal
hold off


data3 = [data1;data2];
theclass = ones(200,1);
theclass(1:100) = -1;

cl = svmtrain(data3,theclass,'Kernel_Function','rbf',...
    'boxconstraint',Inf,'showplot',true);
hold on
axis equal
ezpolar(@(x)1)
hold off


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




