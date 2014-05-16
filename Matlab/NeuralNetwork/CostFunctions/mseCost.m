function [ E ] = mseCost( Y, Hx, reg )
%MSECOST Summary of this function goes here
%   Detailed explanation goes here
     m = size(Y,1);
     E =   1/(2*m) * sum(sum((Y - Hx).^2)) + reg;
          
end

