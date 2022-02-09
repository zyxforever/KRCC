function [I]=Testing_KRCC(E,k,alpha,beta)
 [Kh,Ks,X]=Representation_data(E);
%[Nx,Ny] = size(X);
%[~,cc]=size(E);
[Y,L]=KMeansPlus(X,k);
U=(Label2H(L))';
[I]=KRCC(E,X,k,Y,Kh,Ks,alpha,beta(h),1);
end

function [Kh,Ks,X]=Representation_data(E)
[r,cc]=size(E);
Kh=zeros(1,cc);
Ks=zeros(1,cc+1);
for i=1:cc
    cl{i}=unique(E(:,i));
    Kh(i)=length(cl{i});
    Ks(i+1)=Ks(i)+Kh(i);
end
nc=sum(Kh);
X=zeros(r,nc);
% for j=1:cc
%      X(:,Ks(j)+1:Ks(j+1))=full(sparse([1:r],E(:,j),1));
% end
for j=1:cc
    for i=1:r
        X(i,Ks(j)+find(cl{j}==E(i,j)))=1;
    end
end
end

function [Label]=KRCC(E,X,k,V,Kh,Ks,alpha,beta,lammda)
 [r,nc]=size(X);
[~,cc]=size(E);
% Kh=zeros(1,cc);
% Ks=zeros(1,cc+1);
% for i=1:cc
%     cl{i}=unique(E(:,i));
%     Kh(i)=length(cl{i});
%     Ks(i+1)=Ks(i)+Kh(i);
% end
% nc=sum(Kh);
% if c~=nc
%   Kh=c;
%   Ks=zeros(1,2);
%   Ks(1)=0;
%   Ks(2)=c;
%   nc=c;
%   cc=1;
% end
% Ard=repmat(sum(X),k,1);
U=zeros(k,r);
D=pdist2(X,V);
[~,I]=min(D,[],2);
for i=1:r
    U(I(i),i)=1;
end

%beta=alpha*mean(Kh/k);

%t0=clock;
t=0;

OLDFSS=0;
NEWFSS=1;
W=ones(k,nc)/k;
while OLDFSS~=NEWFSS
    OLDFSS=NEWFSS;
    if t>30
        break;
    end
    t=t+1;
    %Compute V    
    Fri=U*X;
    Fr=diag(1./sum(U,2))*Fri;
    Fr(isnan(Fr))=0;
    %WE=log(W);
    %WE(WE==-Inf)=0;
    %V=(WE*beta-Fr);
    V=1-Fr.*W;
    %V=(1-Fr);
    %alpha=max(std2(V),alpha);
    V=exp(-V/alpha);
    %V(V==exp(-1/alpha))=0;
    %V=V./repmat(sum(V,2),1,nc);
    for j=1:cc
        A=sum(V(:,(Ks(j)+1):Ks(j+1)),2);
        V(:,(Ks(j)+1):Ks(j+1))=diag(1./A)*V(:,(Ks(j)+1):Ks(j+1));
    end
    VE=log(V);
    VE(VE==-Inf)=0;
    %Compute W
    %W=VE*alpha-Fr;
    W=1-Fr.*V;
    %W=(1-Fr);
    %beta=max(std2(W),beta);
    W=exp(-W/beta);
    %W(W==exp(-1/beta))=0;
    W=W./repmat(sum(W),k,1);
    WE=log(W);
    WE(WE==-Inf)=0;
    WE=W.*WE;
    VE=V.*VE;
    %WE=(sum(A,2))';
    %Compute V
    distance=cc-X*(W.*V)'+repmat(lammda*(alpha*(sum(VE,2))'+beta*sum(WE,2)'),r,1);
    [MinValue,MinRow]=min(distance,[],2);
    clear distance
    Label=MinRow;
    %U=sparse(Label,[1:r],1);
    U=zeros(k,r);
     for i=1:r
         U(MinRow(i),i)=1;
     end
    NEWFSS=sum(MinValue);
    NEWFSS=round(NEWFSS*10000)/10000;

end
%times=etime(clock,t0);
end



