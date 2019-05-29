% This is a demo program for the following paper: 
% 
% Fast and Robust Multi-View Multi-Task Learning via Group Sparsity. 
% A submission to the 28th International Joint Conference on Artificial Intelligence. 
%
% The program shows how the propose method, AGILE, copes with a multi-view multi-task dataset.
%
% Please type 'help AGILE_train' or 'help AGILE_test' under MATLAB prompt for more information.

rng('default')
addpath('utils','eval');

%% Data generation for classification (true) or regression (false): 
flagRC  = true;       

%% Generate the synthetic data
agile_initGenSyn;
[data,target,para] = agile_genSyn(para);
data = standardize(data);

%% Set hyper-parameters
agile_initPara;

%% Repeat experiments and save average results
all_res = zeros(nMet,nFold);
for fold_id = 1 : nFold
    [Dtr,~,Dte] = splitData(data,target,para);  % use the left data as Utr
    [Ta,STATS]  = AGILE_train(Dtr.Xl,Dtr.Xu,Dtr.Y,opts);
    [Ypre,Yout] = AGILE_test(Dte.X,Ta,opts);
    [all_res(:,fold_id),met_set] = evaluation(Ypre,Yout,Dte.Y,STATS.time,flagRC);
end
mean_res = squeeze(mean(all_res,2));
std_res  = squeeze(std(all_res,0,2) / sqrt(nFold));

%% Report results
printmat([mean_res,std_res],'SynData_AGILE',met_set,'Mean Std.');