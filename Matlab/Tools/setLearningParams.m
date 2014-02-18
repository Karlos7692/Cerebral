function [LearningParams] = setLearningParams(maxIter, lr, momentum, reg)
%SETTRAININGPARAMS Summary of this function goes here
%   Detailed explanation goes here
    LearningParams.maxIter = maxIter;
    LearningParams.lr = lr;
    LearningParams.momentum = momentum;
    LearningParams.reg = reg;

end

