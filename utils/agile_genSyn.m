function [ data,target,para ] = agile_genSyn( para )
%GENSYN Generate the synthetic multi-view multi-task data for regression or
%classifitcation 
%
% Input
%     para   :   input parameters
%
% Output
%     data   :   heterogeneous data in a cell structure
%     target :   target information in a cell structure
%     para   :   output parameters

%% Get the problem settinga
numN = para.numN;
numT = para.numT;
numV = para.numV;
numD = para.numD;
maxD = para.maxD;
minD = para.minD;
pctD = para.pctD;
th_h = para.thTV1;   % Threshold for useful task-view outliers
th_w = para.thTV2;        % Threshold for useless task-view outlises
flagRC = para.flagRC;

%% Error checking
if th_h>1 || th_h<0 || th_w>1 || th_w<0
    error('Threshold of task-view outliers should be selected from [0,1]');
end

%% Set the parameters for the underlying model and noise
muX  = 0;  sigmaX = 5;
muW  = 0;  sigmaW = 4;
muH  = 0;  sigmaH = 4;
muE  = 0;  sigmaE = 1;

%% Preprocess
vecD   = round(linspace(minD,maxD,numV));
vecDc  = round(vecD*pctD);

%% Determine the pattern of noisy views and task-view outliers
if isfield(para,'mask_h') && isfield(para,'mask_w')
    mask_h = para.mask_h;
    mask_w = para.mask_w;
else
    mask_h = zeros(numT,numV);   
    mask_h(rand(numT,numV)<=th_h) = 1;
    mask_w = ones(numT,numV);
    mask_w(rand(numT,numV)<=th_w) = 0;
    for t = 1 : numT
        for v = 1 : numV
            if mask_h(t,v)==1 && mask_w(t,v)==0
                mask_w(t,v) = 1;
            end
        end
    end
end

%% Expand mask_w and mask_h
mv_diag = cell(numV,1);
for v = 1 : numV
    mv_diag{v} = ones(vecD(v),1);
end
mv_diag = blkdiag(mv_diag{:});
mask_H = mv_diag * mask_h';
mask_W = mv_diag * mask_w';

%% Initialization of data and target
data   = cell(numT,numV);
target = cell(numT,1);

%% Generate the projection matrix P so that P'*P = I
P = cell(numV,1);
for v = 1 : numV
    if v == 1
        useD1 = vecDc(1);
    end
    [Pv,~,~] = svd(rand(vecDc(v),useD1));
    P{v} = Pv(:,1:useD1);
end

%% Generate the matrix W (dual-heterogeneity)
W = [];
for v = 1 : numV
    numDv = vecD(v);
    useDv = vecDc(v);
    difDv = numDv - useDv;
    if v == 1
        W1c = mySample(useDv,numT,muW,sigmaW);
        Wv  = cat(1,W1c,zeros(difDv,numT));
    else
        Wvc = P{v}*W1c;
        Wv  = cat(1,Wvc,zeros(difDv,numT));
    end
    W = cat(1,W,Wv);
end
W = W .* mask_W;

%% Generate the matrix H (Task-view outliers)
H = mySample(numD,numT,muH,sigmaH);
H = H .* mask_H;

%% Produce the training, testing and validation sets
for t = 1 : numT
    for v = 1 : numV
        numDv = vecD(v);
        useDv = vecDc(v);
        difDv = numDv - useDv;
        if v == 1
            data{t,1} = mySample(numN,numDv,muX,sigmaX);            
            data_1c = data{t,1}(:,1:useDv);
        else
            data_vc = data_1c*P{v}';
            data{t,v} = cat(2,data_vc,mySample(numN,difDv,muX,sigmaX));            
        end
        
        if mask_w(t,v) == 0
            data{t,v} = mySample(numN,vecD(v),muX,sigmaX);            
        end
        
    end
end


%% Produce the response Y for training, testing and validation sets
Ta  = W + H;
for t = 1 : numT
    target{t} = (cell2mat(data(t,:))*Ta(:,t))./numV + mySample(numN,1,muE,sigmaE);        
    if flagRC
        target{t} = myBinary(target{t});
    end
end

%% Save results
para.W   = W;
para.H   = H;
para.vecD = vecD;
para.matOw = mask_w;
para.matOh = mask_h;

end


%% Define the sampling function (X \sim N(mu,sigma^2))
function X = mySample(dim1,dim2,mu,sigma)
X = sigma*randn(dim1,dim2)+mu;  
end 

