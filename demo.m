% This is a demo program for the following paper: 
% 
% Fast and Robust Multi-View Multi-Task Learning via Group Sparsity. 
% A submission to the 28th International Joint Conference on Artificial Intelligence. 
%
% The program shows how well does the propose method, AGILE, decompose its parameter model.
%
% Please type 'help AGILE_train' or 'help AGILE_test' under MATLAB prompt for more information.

rng('default')
addpath('utils');

%% Set a designed or random pattern for the weight matrix.
flagDesign   = true;      % designed (true) or random (false)

%% Set parameters
opts.alpha   = 10;        % regularization parameter of l21-norm
opts.beta    = 1;         % regularization parameter of group trace lasso
opts.gamma   = 36;        % regularization parameter of group l1-norm
opts.nIter   = 500;       % maximum number of main iterations
opts.absTol  = 10^-6;     % tolerance for teminating the iterative algorithm
opts.flagEta = 'line';    % 'line' (line search) or 'fix' (fixed value)   
opts.debug   = false;     % show debug information (true) or not (false)

%% Generate the synthetic data
agile_initGenSyn;
[data,target,para] = agile_genSyn(para);
[data] = standardize(data);

%% Split data 
para.trRt = 1.0;          % Ratio of training data
para.teRt = 0.0;          % Ratio of testing data
para.vaRt = 0.0;          % Ratio of validation data
para.pctL = 0.5;          % Ratio of labeled data in the training set
[Dtr,~,~] = splitData(data,target,para);  

%% Train AGILE's model
[Ta,STATS] = AGILE_train(Dtr.Xl,Dtr.Xu,Dtr.Y,opts);

%% Illustration of results
agile_showRes;
