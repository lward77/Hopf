%Kang model with tanh or logistic squashing
%Also includes Wallace-W-C model, Crowley-Martin, Stuart-Landau, and 
%Truscott-Brindley models, all of which have Hopf points
%Sims for behaviour around Hopf point paper
% L Ward 2023-2024-

clear all;
rng='shuffle';
runtime = 1000000;    % run simulation for runtime moments delt
%time step for Kang and tanh Kang: 0.00005, others 0.005 or...
delt=0.00005;    %change re model running
sqrtdelt=delt.^0.5;
t=1:runtime;
dt=delt.*t;   %time running at delt rate

%Kang model parameters %models 0,1,3,8
%SEE=1.8;            %synaptic efficacies - see later for SEE varying
SEI=0.55;      %I on E
SIE=1.0372;
SII=0.1;
%SII=(0.01.*randn(N,1))+(0.1.*ones(N,1)); %freq N(70 Hz, 0.5)
%SII=0.1.*ones(N,1);     %all frequencies = 70 Hz
tauE=0.003;       %time constants
tauI=0.006;
r=1;                  %growth rate of logistic for Kang squashed

%variables for all models
VE=zeros(2,runtime);
VI=zeros(2,runtime);
dVE=zeros(2,runtime);
dVI=zeros(2,runtime);
tempE=zeros(2,runtime);
tempI=zeros(2,runtime);

%VE(:,1)=0.5.*rand(N,1);
%VI(:,1)=0.4.*rand(N,1);
%VE(:,1)=rand(N,1);        %start at random positions
%VI(:,1)=rand(N,1);

%Wallace et al model 2 parameters - need to adjust delt=0.005
alphaE=0.1;                      %active E to quiet
alphaI=0.2;                     %active I to quiet
betaE=1;                        %quiet E to active
betaI=2;                      %quiet I to active
hE=-3.8;        
hI=-9.2;
WEE=20.0;                     % >25 for limit cycles; smaller for quasicycles
WII=1.5;
WIE=32;
WEI=26.3;

%Crowley-Martin parameters model 4; note need to adjust noise 0.02 good - and time
%step delt=0.005
a=1;
a1=2.1;
a2=1.1;
a3=0.001;
b=0.5e-11;   %for fixed point >2 (20 best?), for limit cycle 0.5e-11
c=-0.4;
w=10;
f=5;
r=1;

%Stuart-Landau parameters model 5 - time step delt=0.005 best?
wsl=17.133;       %frequency in radians
bsl=0.0005;         %noise intensity
asl=0;            %bifurcation parameter

%Truscott–Brindley model 6 parameters from Bashkirtseva-Ryashko delt=0.5
dtb = 0.1;                
rtb = 1.0;
ktb = 1.0;
mtb = 1.0;
btb = 7.0;
alpha=2.0;

%FitzHugh-Nagumo model 7 delt=0.005 Hadeler etal a=1, b=.5, gamma=2
aFHN=0.14;%1.0;%0.2; Baer et al values 0.2, 0.5, 0.4
bFHN=0.008;%0.5;%0.05; Kuske values 0.14, 0.008, 2.54
gFHN=2.54;%2.0;%0.4;

%BG 1D type tanh model 9
epsilon=0.0001;   %this parameter controls size of lambda=epsilon.*dt
lambda=zeros(2,runtime);

model=8;        %set model type: 0=logistic Kang, 1=clipped Kang, 2=Wallace-W-C, 3=tanh Kang,
                %4=Crowley-Martin from Lv et al., 5=Stuart-Landau, 6=Truscott-Brindley, 
                %7=FitzHugh-Nagumo, 8=simple linear Kang 9=BG-type 1D tanh
                %or -x^3 model -remember to reset other parameters

%starting values for VE and VI
startVE=0.001;
startVI=0.001;
VE(1,1)=startVE;      
VI(1,1)=startVI;
VE(2,1)=startVE;       
VI(2,1)=startVI;                
                
                %set noise intensities for chosen model
sigE=0.002;%00000002;      
sigI=0.002;%00000002;
                               
%set range for critical parameter

SEE=zeros(2,runtime);    %Kang models 0,1,3,8
SEE(:,1)=1.45;
SEEinc=0.000;

WEE=zeros(2,runtime);    %Wallace W-C model 2
WEE(:,1:2)=22;
WEEinc=0.006;

b=zeros(2,runtime);      %Crowley-Martin model from Lv et al. 4
b(:,1:100001)=0.05;%0.5e-11;%20;
binc=0.85;

asl=zeros(2,runtime);     %Stuart-Landau model 5 from Dagnino,Laureys,Deco
asl(:,1:2)=-0.5;
aslinc=0.000000;

atb=zeros(2,runtime);     %Truscott–Brindley model 6  
atb(:,1:2)=6.5;
atbinc=0.000001;

Ioft=zeros(2,runtime);   %FitzHugh-Nagumo model 7
Ioft(:,1:2)=0.05;        %Kuske/Baer have two I: I_SH=0.04 and slowly
Ioftinc=0.0003;          %changing I_SS or constant I_D



for j=1:2               %noise loop beginning
    if j==1
        nse=0;
    else
        nse=1;
    end

for jj=1:999:runtime   %change depending on which model is running
    %and parameter change speed
    %note increment same as that for critical parameter range above
    %SEE(j,jj)=t(1,jj)^0.02+0.3;
    SEE(j,jj)=SEE(j,jj)+SEEinc;     %model 0,1,3 or 8
    %WEE(j,jj)=WEE(j,jj)+WEEinc;     %model 2
    %b(j,jj)=b(j,jj).*binc;          %model 4
    %asl(j,jj)=asl(j,jj)+aslinc;     %model 5
    %atb (j,jj)=atb(j,jj)+atbinc;    %model 6
    %Ioft(j,jj)=Ioft(j,jj)+Ioftinc;   %model 7
    for kk=jj:jj+999   %change depending on model and parameter change speed
        SEE(j,kk)=SEE(j,jj);         %model 0,1,3 or 8
        %WEE(j,kk)=WEE(j,jj);        %model 2
        %b(j,kk)=b(j,jj);            %model 4
        %asl(j,kk)=asl(j,jj);        %model 5
        %atb(j,kk)=atb(j,jj);        %model 6
        %Ioft(j,kk)=Ioft(j,jj);       %model 7
    end
    %lambda(j,jj)=epsilon.*dt(1,jj);    %model 9 - note same as d lambda=epsilon dt solved
end    

kns=1.0;
for k=2:runtime      %basic loop
    
    noiseE=nse.*randn(1,1);      %independent noises
    noiseI=nse.*randn(1,1);    %nse=0 for deterministic
    %SEE(j,k)=SEE(j,k);%+kns.*randn(1,1);
    %now compute voltages - solve SDE using Euler-Maruyama
    %squashed where r is growth rate of logistic, middle at 0

    if model==0 %kang logistic squashed
    dVE(j,k)=(delt.*(-VE(j,k-1)+(1./(1+(exp(-r.*((+SEE(j,k).*VE(j,k-1))-(SEI.*VI(j,k-1))))))))+(sigE.*sqrtdelt.*noiseE))./tauE; 
    dVI(j,k)=(delt.*(-VI(j,k-1)+(1./(1+(exp(-r.*((-SII.*VI(j,k-1))+(SIE.*VE(j,k-1))))))))+(sigI.*sqrtdelt.*noiseI))./tauI; 
    VE(:,k)=VE(j,k-1)+dVE(j,k);
    VI(:,k)=VI(j,k-1)+dVI(j,k);

    else if model==1        %Kang clipped sort of like MGW
    dVE(j,k)=(delt.*(-VE(j,k-1)+((+SEE(j,k).*VE(j,k-1))-(SEI.*VI(j,k-1)))+(sigE.*sqrtdelt.*noiseE)))./tauE; 
    dVI(j,k)=(delt.*(-VI(j,k-1)+((-SII.*VI(j,k-1))+(SIE.*VE(j,k-1)))+(sigI.*sqrtdelt.*noiseI)))./tauI; 
    
    if (VE(j,k-1)+dVE(:,k)) gt 1 || (VE(j,k-1)+dVE(j,k))  lt -1
        VE(j,k)=VE(j,k-1);        
    else
        VE(j,k)=VE(j,k-1)+dVE(j,k);        
    end

    if (VI(j,k-1)+dVI(j,k))>1 || (VI(j,k-1)+dVI(j,k)) <-1
        VI(j,k)=VI(j,k-1);
    else
        VI(j,k)=VI(j,k-1)+dVI(j,k);
    end

    elseif model==2  %Wallace et al Wilson-Cowan
    dVE(j,k)=delt.*(alphaE.*(-VE(j,k-1))+(1-VE(j,k-1)).*betaE.*(1./(1+exp(-r.*((+(WEE(j,k).*VE(j,k-1))-(WEI.*VI(j,k-1))+hE))))));
    dVI(j,k)=delt.*(alphaI.*(-VI(j,k-1))+(1-VI(j,k-1)).*betaI.*(1./(1+exp(-r.*((-(WII.*VI(j,k-1))+(WIE.*VE(j,k-1))+hI))))));
    VE(j,k)=VE(j,k-1)+dVE(j,k)+(sigE.*sqrtdelt.*noiseE);
    VI(j,k)=VI(j,k-1)+dVI(j,k)+(sigI.*sqrtdelt.*noiseI);

    elseif model==3 %Kang with tanh squashing
    tempE(j,k)=tanh((+SEE(j,k).*VE(j,k-1))-(SEI.*VI(j,k-1)));
    tempI(j,k)=tanh((-SII.*VI(j,k-1))+(SIE.*VE(j,k-1)));
    dVE(j,k)=(delt.*(-VE(j,k-1)+tempE(j,k))+(sigE.*sqrtdelt.*noiseE))./tauE; 
    dVI(j,k)=(delt.*(-VI(j,k-1)+tempI(j,k))+(sigI.*sqrtdelt.*noiseI))./tauI; 
    VE(j,k)=VE(j,k-1)+dVE(j,k);
    VI(j,k)=VI(j,k-1)+dVI(j,k);

    elseif model==4    %Crowly-Martin predator prey model from Lv et al 2020
    dVE(j,k)=(r-(a.*VE(j,k-1))-((w.*VI(j,k-1))./(1+a1.*VE(j,k-1)+a2.*VI(j,k-1)+a3.*VE(j,k-1).*VI(j,k-1)))).*VE(j,k-1).*delt+(sigE.*0.05.*VE(j,k-1).*noiseE);
    dVI(j,k)=(c-(b(j,k).*VI(j,k-1))+((f.*VE(j,k-1))./(1+a1.*VE(j,k-1)+a2.*VI(j,k-1)+a3.*VE(j,k-1).*VI(j,k-1)))).*VI(j,k-1).*delt+(sigI.*0.03.*VI(j,k-1).*noiseI);
    VE(j,k)=VE(j,k-1)+dVE(j,k);
    VI(j,k)=VI(j,k-1)+dVI(j,k);

    elseif model==5        %Stuart-Landau from various, esp Deco-Laureys
    dVE(j,k)=((((asl(j,k-1)-VE(j,k-1)^2-VI(j,k-1)^2).*VE(j,k-1))-wsl.*VI(j,k-1)).*delt)+bsl.*noiseE;
    dVI(j,k)=((((asl(j,k-1)-VE(j,k-1)^2-VI(j,k-1)^2).*VI(j,k-1))+wsl.*VE(j,k-1)).*delt)+bsl.*noiseI;
    VE(j,k)=VE(j,k-1)+dVE(j,k);
    VI(j,k)=VI(j,k-1)+dVI(j,k);

    elseif model==6         %Truscott–Brindley model
    dVE(j,k)=(((rtb.*VE(j,k-1).^alpha).*((ktb+sigE.*noiseE)-VE(j,k-1))-((atb(j,k-1).^2.*VE(j,k-1).^2)./(1+btb.^2.*VE(j,k-1).^2)).*VI(j,k-1))./dtb).*delt;
    dVI(j,k)=(((atb(j,k-1).^2.*VE(j,k-1).^2)/(1+btb.^2.*VE(j,k-1).^2))-mtb.*VI(j,k-1)).*delt;
    VE(j,k)=VE(j,k-1)+dVE(j,k);
    VI(j,k)=VI(j,k-1)+dVI(j,k);

    elseif model==7         %FitzHugh-Nagumo model VE is v, VI is w
    dVE(j,k)=(-VE(j,k-1)^3+((VE(j,k-1)^2).*(aFHN+1))-aFHN.*VE(j,k-1)-VI(j,k-1)+Ioft(j,k)).*delt+(1.41421.*sigE.*noiseE);
    dVI(j,k)=(bFHN.*(VE(j,k-1)-gFHN.*VI(j,k-1))).*delt;
    VE(j,k)=VE(j,k-1)+dVE(j,k);
    VI(j,k)=VI(j,k-1)+dVI(j,k);
    
    elseif model==8          %Kang standard - note linear - no limit cycles - blows up for SEE >1.55
    dVE(j,k)=(delt.*(-VE(j,k-1)+((+SEE(j,k).*VE(j,k-1))-(SEI.*VI(j,k-1)))+(sigE.*sqrtdelt.*noiseE)))./tauE; 
    dVI(j,k)=(delt.*(-VI(j,k-1)+((-SII.*VI(j,k-1))+(SIE.*VE(j,k-1)))+(sigI.*sqrtdelt.*noiseI)))./tauI; 
    VE(j,k)=VE(j,k-1)+dVE(j,k);
    VI(j,k)=VI(j,k-1)+dVI(j,k);

    elseif model==9         %1D model after BG
    dVE(j,k)=(tanh(lambda(j,k).*VE(j,k-1)).*delt)+(sigE./sqrt(epsilon).*noiseE);
    %dVE(j,k)=((lambda(j,k).*VE(j,k-1)-VE(j,k-1).^3).*delt)+(sigE./sqrt(epsilon).*noiseE);
    VE(j,k)=VE(j,k-1)+dVE(j,k);
    end
    end
end    %end of k loop
end    %end of j (noise) loop

hVE1=hilbert(VE(1,:));
hVI1=hilbert(VI(1,:));
ampVE1=abs(hVE1);
ampVI1=abs(hVI1);
phVE1=angle(hVE1);
phVI1=angle(hVI1);

hVE2=hilbert(VE(2,:));
hVI2=hilbert(VI(2,:));
ampVE2=abs(hVE2);
ampVI2=abs(hVI2);
phVE2=angle(hVE2);
phVI2=angle(hVI2);

npchange=1000;      % redimension re number of parameter changes
A=zeros(1,npchange);   
B=zeros(1,npchange);
indx=1;
inA=ones(1,npchange);
for kkk=1:1000:runtime-1 %change inA depending on model and parameter change speed
    inA(1,indx)=Ioft(1,kkk);   %log10(b(1,kkk));%log10 for Crowley b, not for SEE,WEE,asl,Ioft
    if inA(1,indx) == 0
        inA(1,indx)=Ioft(1,kkk);
    end
    A(1,indx)=mean(ampVE2(1,kkk:kkk+999)); %step size=time parameter same
    B(1,indx)=mean(ampVI2(1,kkk:kkk+999));
    indx=indx+1;
end

[fE,gofE]=fit(inA',A','poly2');
%[fI,gofI]=fit(inA',B','poly4');
f1=figure;hold on;
scatter(inA,A);
%scatter(inA,B);
plot(fE,'b');
%plot(fI,'r');
xlabel('Ioft value');       %change for other models
ylabel('mean Gabor amplitude');
title('Stochastic Cycle Amplitude');
%legend('Gabor ampVE','Gabor ampVI');
gofE
fE
%gofI
%fI
AA=A(1,1:550);
BB=(1.5215-inA);

%power spectrum
Nt=10000;
Fs=1./delt;   %sampling frequency
%[pxx1,fx1]=pmtm(VE(1,Nt:runtime),100,length(VE(1,Nt:runtime)),Fs);
[pxx2,fx2]=pmtm(VE(2,Nt:runtime),100,length(VE(2,Nt:runtime)),Fs);
f2=figure;hold on;
%plot(log10(fx1(1:4000)),log10(pxx1(1:4000)),'b');
plot(log10(fx2(1:30000)),log10(pxx2(1:30000)),'r');
title('Power Spectrum');
xlabel('Frequency log Hz');
ylabel('Log Spectral Power');
%legend('Deterministic','Noisy');

f2=figure;hold on;    %drop initial Nt time points
Nt=1;
timediv=1./delt;    %time points per sec
plot3(t(1,Nt:runtime)./timediv,VE(1,Nt:runtime),VI(1,Nt:runtime),'b');
plot3(t(1,Nt:runtime)./timediv,VE(2,Nt:runtime),VI(2,Nt:runtime),'g');
%plot(t(1,Nt:runtime),VE(1,Nt:runtime),'b');
%plot(t(1,Nt:runtime),VE(2,Nt:runtime),'g');
title(['Phase plot over time']);
xlabel('Time (sec)');
ylabel('VE');
zlabel('VI');
legend('Deterministic','Noisy');

f3=figure;hold on;%change x axis depending on model
plot3(SEE(1,Nt:runtime),VE(1,Nt:runtime),VI(1,Nt:runtime),'b');
title(['Phase plot over parameter']);
%xlabel('I');
%ylabel('VE');
%zlabel('VI');
%f4=figure;
plot3(Ioft(1,Nt:runtime),VE(2,Nt:runtime),VI(2,Nt:runtime),'g');
%plot(SEE(1,Nt:runtime),VE(1,Nt:runtime),'b');
%plot(SEE(1,Nt:runtime),VE(2,Nt:runtime),'g');
title(['Phase plot over parameter']);
xlabel('SEE');
ylabel('VE');
zlabel('VI');
legend('Deterministic','Noisy');


%f4=figure;hold on;
%for l=1:10000:runtime
 %   plot3(t(1,l:l+1000),VE(1,l:l+1000),VI(1,l:l+1000),'b');
  %  plot3(t(1,l:l+1000),VE(2,l:l+1000),VI(2,l:l+1000),'r');
%end
%title('Phase plot over time');
%xlabel('Time and SEE (1.4 to 1.697)');
%ylabel('VE');
%zlabel('VI');

mga=mean(ampVE2(1,50000:100000));

f5=figure;
plot3(t(1,Nt:200000)./timediv,VE(1,Nt:200000),VI(1,Nt:200000),'b');
title(['Phase plot over time']);
xlabel('Time (sec)');
ylabel('VE');
zlabel('VI');