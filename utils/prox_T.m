function [P_hat,P_tn] = prox_T(L,alpha)
% Singular Value Thresholding
% min_P 0.5*||L-P||_F^2 - alpha*||P||_*

[numR,numC] = size(L);

% Produce the 'Economy Size' SVD
if (numR > numC)
    [M,S,N] = svd(L, 0);
    
    valTh = diag(S) - alpha;   
    diag_S = valTh .* ( valTh > 0 );
    P_hat = M * sparse(diag(diag_S)) * N';
    P_tn = sum(diag_S);
else 
    [M,S,N] = svd(L', 0);
    
    valTh = diag(S) - alpha;   
    diag_S = valTh .* ( valTh > 0 );
    
    P_hat = M * sparse(diag(diag_S)) * N';
    P_hat = P_hat';
    P_tn = sum(diag_S);
end
