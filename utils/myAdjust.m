function [ A ] = myAdjust( A, min_val, max_val )
%MYADJUST Summary of this function goes here
%   Detailed explanation goes here

A(abs(A)<mean(abs(A(:)))) = min_val;
A(abs(A)>0) = max_val;

end

