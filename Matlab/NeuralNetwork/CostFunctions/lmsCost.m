function [ E ] = lmsCost(Y, Hx, reg)
%LMSCOST Summary of this function goes here
%   Detailed explanation goes here
    E =   1/2 * sum(sum((Y - Hx).^2)) + reg;

end

