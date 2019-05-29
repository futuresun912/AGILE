function [W] = prox_L21(W_hat, tau)
% min_X 0.5*||X - D||_F^2 + tau*||X||_{1,2}
% where ||X||_{1,2} = sum_i||X^i||_2, where X^i denotes the i-th row of X
W = repmat(max(0, 1 - tau./sqrt(sum(W_hat.^2,2))),1,size(W_hat,2)).*W_hat;

end