function [Fj, fAll, R2, actInd] = ...
                      estimateAdditiveRKHS(Y,features, B, maxIter, nVar)

%Estimates the regression in a RHKS where the covariance kernel is
%implicitly given by the features: features(:,:,k)*features(:,:,k)' is the  
% covariance kernel evaluated at the sample points for the k^th variable.                   
% Y is the dependent variable. B is the size of the ball of the space L(B)
% which is the space of additive functions such that the sum of the RKHS
% norm of each additive component is less or equal than B.
% maxIter is the max number of iterations of the greedy algorithm
% nVar is the number of active variables to be seleceted before 
% the algorithm stops, unless maxIter is reached before. 

%The function outputs:
% Fj: the fitted valued 
% fJ: the fitted values for each component
%R2: the R square
%actInd: the variable chosen at each iteration 

% For more details:
% Sancetta, A. (2017) Inference for Additive Models in the Presence of Possibly
% Infinite Dimensional Nuisance Parameters. URL:https://arxiv.org/abs/1611.02199

if nargin <3
    
    B = 1;
    
end

if nargin <4
    
   maxIter = 500;
    
end

if nargin <5
    
    
    nVar = size(features,3);
    
end

coefAll          = zeros(maxIter,size(features,2)); 
Fj               = 0;
fAll             = zeros(size(features,1),size(features,3));
obj              = 0;
j                = 1;
nActive          = 0;
actInd           = zeros(maxIter,1);
stop             = false;

while (j<=maxIter) && (nActive<= nVar) && ~stop



grad             = -2*(Y-Fj);
[fj, sj, aj]     = get_maxFunc(grad,features);

%% line search
fj               = fj*B;
aj               = aj*B;
func             = @(gamma) objective(fj,Fj,Y,gamma);
gamma            = fminbnd(func,0,1);
fj               = gamma*fj;
aj               = gamma*aj;
Fj               = (1-gamma)*Fj+fj;
fAll             = (1-gamma)*fAll;
fAll(:,sj)       = fAll(:,sj)+fj;
coefAll          = (1-gamma)*coefAll;
coefAll(j,:)       = aj;

actInd(j)        = sj;
j                = j+1;
nActive          = length(unique(actInd));
stop             = abs(func(gamma)-obj)<1e-6*obj ;
obj              = func(gamma);

end
  
if (nActive> nVar)
fAll(:,sj)      = fAll(:,sj)-fj;
fAll            = fAll/(1-gamma);
Fj              = sum(fAll,2);
end    

R2              = 1-(Y-Fj)'*(Y-Fj)/var(Y)/length(Y);

end




function out = objective(funj,F,Y,gamma)

out          = (Y-(1-gamma)*F-gamma*funj);
out          = out'*out/length(Y);

end


function [outFunc, outInd, outCoef] = get_maxFunc(grad,features)

[n L K]          = size(features);
norms            = zeros(1,K);    
for k =1:K
norms(k)            = sum((grad'*features(:,:,k)/n).^2);    
end

[maxNorm, outInd] = max(norms);

outCoef          = -1*(grad'*features(:,:,outInd)/n);
outFunc          = features(:,:,outInd)*outCoef';
if maxNorm>0
    outFunc          = outFunc/sqrt(maxNorm);
    outCoef          = outCoef/sqrt(maxNorm);
else
    outFunc          = outFunc*0;
    outCoef          = outCoef*0;
    
end

end
