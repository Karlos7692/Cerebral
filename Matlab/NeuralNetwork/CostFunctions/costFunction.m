function [ E, gradconst ] = costFunction(Y, Hx, reg, type)
%COSTFUNCTION Summary of this function goes here
%   Detailed explanation goes here
    gradconst = 1;
    if strcmp(type, 'lms')
        E = lmsCost(Y, Hx, reg);    
    elseif strcmp(type, 'mse')
        E = mseCost(Y, Hx, reg);
        gradconst = 1/size(Y,1);
    else
        
    end
        

end

