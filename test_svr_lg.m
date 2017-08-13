
trainnum=1000;
pi=3.1415926;
xs=0;
xe=20*pi;
noise_m=0;
noise_d=0.1;
mx=xs:0.1:xe;
sinx=sin(mx);
x = [xs + (xe-xs)/2.*rand(trainnum/2,1) ; xs+(xe-xs)/2 + (xe-xs)/2.*rand(trainnum/2,1) ]
y= [sin(x(1:trainnum/2))+ noise_m + noise_d.*randn(trainnum/2,1);   2* sin(x(trainnum/2+1:trainnum))+ noise_m + noise_d.*randn(trainnum/2,1) ];


xs=-2*pi;
xe=22*pi;
xt = xs:0.1:xe;% + (xe-xs).*rand(trainnum,1);
xt=xt';
testnum=size(xt,1);
yt= [sin(xt(1:testnum/2));   2* sin(xt(testnum/2+1:testnum)) ];




model = svmtrain(y,x,'-s 3 -t 2 -c 2.2 -g 2.8 -p 0.01');
[py,mse,pe] = svmpredict(yt,xt,model);
figure;
plot(x,y,'o');
hold on;
plot(xt,py,'r-','LineWidth',3);


 p3=polyfit(x,y,50);
 f3 = polyval(p3,xt);
 p9=polyfit(x,y,150);
 f9 = polyval(p9,xt);
 hold on;
 plot(xt,f3,'g-',xt,f9,'k-','LineWidth',3);
 legend('train','svr','poly9','poly19');
  axis([xs,xe,-2,2]);


