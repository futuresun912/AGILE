function [ H ] = prox_LG1( H_hat, tau, idxD )
%PROXL2NORM Proximal operator for the summation fo L2-norms
% min_ht 0.5*|| ht - rt ||_2  + beta*sum_v||ht^v||_2

%% Thresholding
myProx = @(x) max(0,1-tau/sqrt(sum(x.^2))) .* x;

%% Proximal operation on each view
[numD,numT] = size(H_hat);
numV = length(idxD);
H = zeros(numD,numT);
for t = 1 : numT
    for v = 1 : numV
        H(idxD{v},t) = myProx(H_hat(idxD{v},t));
    end
end

end

