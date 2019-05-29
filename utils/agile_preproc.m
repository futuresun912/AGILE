function [dX,dU,cY,Utmp,stats] = agile_preproc(X,U,Y)
%PRECALRES diagonalize the input data, and concatenate the target
% dX : diagonalized X
% dU : diagonalized U
% cY : concatenated Y

%% Diagonalize data and concatenate the target
[numT,numV] = size(X);
vecM = zeros(numT,1);
dX = cell(numT,1);
Utmp = cell(numT,1);
for t = 1:numT  
    vecM(t) = size(U{t,1},1);
    dX{t} = cell2mat(X(t,:));
    Utmp{t} = sparse(blkdiag(U{t,:}));
end
dX = sparse(blkdiag(dX{:})) ./ numV;
dU = sparse(blkdiag(Utmp{:}));
cY = cell2mat(Y);

%% Save the view idx
vecD = zeros(numV,1);
idxD = cell(numV,1);  
startD_id = 1;
for v = 1:numV
    vecD(v) = size(X{1,v},2);
    endD_id = startD_id + vecD(v);
    idxD{v} = startD_id : (endD_id-1);
    startD_id = endD_id;
end

%% Save data statistic
stats.numD = sum(vecD);
stats.numT = numT;
stats.numV = numV;
stats.vecM = vecM;
stats.idxD = idxD;

end