function [data] = standardize(data)
% Standardize the input data and add bias dimensionality to each view

[numT,numV] = size(data);
for t = 1 : numT
    numNt = size(data{t,1},1);
    for v = 1 : numV
        data{t,v} = zscore(data{t,v});
        data{t,v} = cat(2,data{t,v},ones(numNt,1));
    end
end

end