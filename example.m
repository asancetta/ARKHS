%------------------------ example of code use

%----------simulate some features for K variables

n          = 1000; %sample size
K          = 10; % number of covariates
p          = 10;%order of polynomial
lambda     = 1.1;%lambda as in paper 
signal2noise = 1;%signal to noise of true model 
X          = 2*rand(n,K)-1;%covariates in [-1,1];


% function to generate polynomial features: user needs to write their own
% features. Note that features for each covariate has a small intercept, so that
% covariance kernel is strictly positive. 
polyFeatures  = @(x) [0.1*ones(n,1),...
                      bsxfun(@rdivide, bsxfun(@power, x, (1:p)), (1:p).^lambda)];
 

% loop to construct array of features
features      = zeros(n,1+p,K); 
for k=1:K
   
    features(:,:,k)  = polyFeatures(X(:,k));
    
end

%NOTE: features(:,:,k)*features(:,:,k)'  is the k^th covariance kernel evaluated
%at the sample points 

% suppose true model is linear with slope one and zero intercept
% extract features under the null (the null is true) and under alternative
mu0                   = zeros(n,1); 
features1             = features(:,3:end,:);
features0             = features(:,1:2,:);%include intercept 

for k=1:K
   
    mu0               = mu0+features(:,2,k);% only pick up the linear feature
    
end

varMu0     = var(mu0);

error      = randn(n,1)*sqrt(varMu0/signal2noise); 
Y          = mu0+error;  

%-------- estimate the model under the null setting B = 10*std(Y) for simplicity; 
B                      = 10*std(Y);
[Fj, fAll, R2, actInd] = ...
                      estimateAdditiveRKHS(Y,features0, B);

error_hat              = Y-Fj;

% -------------------carry out the test

% the features can be reshaped into a matrix.
[n,L,K]      = size(features1);
[n,L0,K0]    = size(features0);

KL           = K*L;
KL0          = K0*L0;

features1   = reshape(features1,n,KL);
features0   = reshape(features0,n,KL0);
                                                                
% eliminate any zero columns and the constant if not interested in testing
% for a constant features
indNoZeroOne1   = ~(all(features1==0,1) |all(features1==1,1));
indNoZeroOne0   = ~(all(features0==0,1) |all(features0==1,1));
Gamma0          = features0(:,indNoZeroOne0);
Gamma1          = features1(:,indNoZeroOne1);


rho    =  0.01*log(1+n)/sqrt(n);
nSim  = 10000;
useFeaturesMap1 = 1;%    
% consider the case based on instruments Gamma1*Gamma1' scaled to unit RKHS
% norm equal to one and non-zero rho
tic
out= testStatProjection(error_hat,Gamma1, Gamma0, rho, nSim, useFeaturesMap1);                                                                
toc                  
% use default
out1= testStatProjection(error_hat,Gamma1, Gamma0);                                                                
                                                                
                                                                
                                                                
                                                                
                                                                
                                                                
                                                                
                                                                
                                                                
                                                                
                                                                
                                                                