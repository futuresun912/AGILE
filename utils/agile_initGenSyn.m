%INITGENSYN Initialization of generation of synthetic datasets
if ~exist('flagRC','var')
    flagRC = false;
end
if ~exist('flagDesign','var')
    flagDesign = false;  % Designed pattern (true) or randome pattern (false)
end

%% Set parameters for the synthetic data
ratioD      = 3;         % ratioD = maxD / minD
para.flagRC = flagRC;    % classification (true) or regression ('false') task
para.numT   = 12;        % number of tasks
para.numV   = 6;         % number of views
para.numN   = 300;       % number of samples 
para.numD   = 300;       % number of features
para.pctD   = 0.3;       % percentage of useful features

%% Calculate the minimium and maximum number of features
[para.minD,para.maxD,para.numD] = allocViews(para.numD,para.numV,ratioD);

%% For generateSyn (currently used one)
para.thTV1  = 0.1;       % percentage of task outliers and inconsistent views in H
para.thTV2  = 0.0;       % percentage of noisy views in W

%% Design the pattern for thTV1
if flagDesign
    mask_h = cell(para.numV,1);
    for i = 1 : para.numV
        mask_h{i} = [1;1];
    end
    mask_h = blkdiag(mask_h{:});
    para.mask_h = mask_h;
    mask_tmp = fliplr(mask_h);
    mask_w = ones(size(mask_tmp));
    mask_w(mask_tmp==1) = 0;
    para.mask_w = mask_w;
end