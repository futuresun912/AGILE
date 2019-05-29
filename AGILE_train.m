function [ Ta,STATS ] = AGILE_train( X,U,y,opts )
%AGILETRAIN Training program for AGILE
%
%       Input:
%           X       T x V labeled data cell, each entry denotes a labeled data matrix of a specific task-view
%           U       T x V unlabeled data cell, each entry denotes a unlabeled data matrix of a specific task-view
%           Y       T x 1 target cell, each entry is a target vector of a specific task-view
%           opts    hyper-parameters of AGILE
%
%       Output
%           Ta      d x T weight matrix, each column is a weight vector for a specific task (Ta = W + H)
%           STATS   a structure containing the learned models and information

%% Set a timer
Tstart = cputime;

%% Get input parameters
alpha = opts.alpha;
beta  = opts.beta;
gamma = opts.gamma;
nIter = opts.nIter;
absTol = opts.absTol;
flagEta = opts.flagEta;
switch flagEta
    case 'fix'
        eta = opts.eta;
    case 'line'
        eta = 1;
end

%% Preprocess the dataset
[X,U,y,Utmp,statD] = agile_preproc(X,U,y);

%% Problem setting
numT = statD.numT;
numV = statD.numV;
numD = statD.numD;
idxD = statD.idxD;
vecM = statD.vecM;

%% Initialization
W = zeros(numD,numT);
H = zeros(numD,numT);
a_new = 1;
W_hat = W;
H_hat = H;
Fval = [];  
P = cell(numT,1);
for t = 1 : numT
    P{t} = zeros(vecM(t),numV);
end
P_hat = P;
P_old = P;
Q     = P;
Q_hat = P;
Q_old = P;
UW = cell(numT,1);

%% Iterative optimization 
for iter = 1 : nIter     
    % 0. Initialization for accelerated algorithm
    W_old = W;
    H_old = H;
    for t = 1 : numT
        Q_old{t} = Q{t};
        P_old{t} = P{t};
    end
    a_old = a_new;
    
    % 1. Calculate z
    z = [];
    for t = 1 : numT
        z = cat(1,z,P_hat{t}(:)-Q_hat{t}(:));
    end
    
    % 2. Update W and H
    grad_H = X' * (X*(W_hat(:)+H_hat(:))-y);
    grad_W = grad_H + U'*(U*W_hat(:)-z);
    grad_H = reshape(grad_H,numD,numT);
    grad_W = reshape(grad_W,numD,numT);
    switch flagEta
        case 'fix'
            W = prox_L21(W_hat-eta.*grad_W,eta*alpha);
            H = prox_LG1(H_hat-eta.*grad_H,eta*gamma,idxD);
        case 'line'
            while 1
                W = prox_L21(W_hat-eta.*grad_W,eta*alpha);
                H = prox_LG1(H_hat-eta.*grad_H,eta*gamma,idxD);
                dw = W(:) - W_hat(:);  
                dh = H(:) - H_hat(:);
                Xw = X * dw;
                Xh = X * dh;
                Uw = U * dw;
%                 Xt = X * (dw+dh);
%                 if eta*(Xt'*Xt+rho*(Uw'*Uw)) <= dw'*dw+dh'*dh                
                if eta*(Xw'*Xw+Uw'*Uw+Xh'*Xh) <= dw'*dw+dh'*dh
                    break;
                else
                    eta = eta / 2;
                end
            end
    end
    
    % 3. Update P and Q
    for t = 1 : numT
        UW{t} = reshape(Utmp{t}*W(:,t),vecM(t),numV);
        P{t}  = prox_T(UW{t}+Q_hat{t},beta);
        Q{t}  = Q_hat{t} + UW{t} - P{t};
    end
    
    % 4. Convergence analysis
    Fval = cat(2,Fval,calFval(X,UW,y,W,H,opts,statD));
    if iter > 1
        if abs(Fval(end,end)-Fval(end,end-1))<absTol || ...
                abs(Fval(end,end)-Fval(end,end-1))/Fval(end,end-1)<absTol || ...
                abs(Fval(end,end))/abs(Fval(end,end-1))>1.1
            break;
        end
    end

    if opts.debug
        disp(['The function value of the ',num2str(iter),'th iteration is ',num2str(Fval(end)),'.']);
    end
    
    % 5. Update coefficients
    a_new = (1+sqrt(1+4*a_old^2))/2;
    W_hat = W + (a_old-1)/a_new*(W-W_old);
    H_hat = H + (a_old-1)/a_new*(H-H_old);
    for t = 1 : numT
        P_hat{t} = P{t} + (a_old-1)/a_new*(P{t}-P_old{t});
        Q_hat{t} = Q{t} + (a_old-1)/a_new*(Q{t}-Q_old{t});
    end
    
end
if opts.debug
    disp(['AGILE converged at the ',num2str(iter),'th iteration with ',num2str(Fval(end)),'.']);
end

%% Save results
Ta = W + H;
STATS.W = W;  
STATS.H = H; 
STATS.Fval = Fval;
STATS.time = cputime - Tstart;

end

%% Calculate the objective value
function fval = calFval(X,UW,y,W,H,opts,stats)

numT = stats.numT;
numV = stats.numV;
idxD = stats.idxD;
fval_tmp1 = 0;
fval_tmp2 = 0;
for t = 1 : numT
    fval_tmp1 = fval_tmp1 + sum(svd(UW{t},0));
    for v = 1 : numV
        fval_tmp2 = fval_tmp2 + sqrt(sum(H(idxD{v},t).^2));
    end
end

fval = zeros(5,1);
fval(1) = 0.5 * sum((y-X*(W(:)+H(:))).^2);        
fval(2) = opts.alpha * sum(sqrt(sum(W.^2,2)));  
fval(3) = opts.beta  * fval_tmp1;
fval(4) = opts.gamma * fval_tmp2; 
fval(5) = sum(fval);

end
