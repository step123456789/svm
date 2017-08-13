


trainnum1=200;
trainnum2=200;
usedtrain=100;
m1=0;
d1=1;
m2=2;
d2=1;
xs=m1-4*d1;
xe=m2+4*d2;
% 
x1=m1 +d1.*randn(trainnum1,2);
x2=m2 +d2.*randn(trainnum2,2);

y=[zeros(trainnum1,1);ones(trainnum2,1)];
x=[x1;x2];

ytrain=[zeros(usedtrain,1);ones(usedtrain,1)];
xtrain=[x1(1:usedtrain,:);x2(1:usedtrain,:)];

b=glmfit(xtrain,ytrain,'binomial','link','logit');

figure;
[ax1,ax2]=meshgrid(xs:0.1:xe);
ax1=reshape(ax1,size(ax1,1)*size(ax1,2),1);
ax2=reshape(ax2,size(ax2,1)*size(ax2,2),1);

yfit = glmval(b, [ax1 ax2],'logit');
yfit=yfit>0.5;
scatter(ax1,ax2,[],yfit,'filled');
hold on
plot(x1(:,1),x1(:,2),'b+',x2(:,1),x2(:,2),'ro');
hold on
plot(x1(1:usedtrain,1),x1(1:usedtrain,2),'c+',x2(1:usedtrain,1),x2(1:usedtrain,2),'yo');


figure;
SVMstruct = svmtrain(xtrain,ytrain,'Kernel_Function','linear');
newClasses = svmclassify(SVMstruct,[ax1 ax2]);
scatter(ax1,ax2,[],newClasses,'filled');
hold on
plot(x1(:,1),x1(:,2),'b+',x2(:,1),x2(:,2),'ro');
hold on
plot(x1(1:usedtrain,1),x1(1:usedtrain,2),'c+',x2(1:usedtrain,1),x2(1:usedtrain,2),'yo');




