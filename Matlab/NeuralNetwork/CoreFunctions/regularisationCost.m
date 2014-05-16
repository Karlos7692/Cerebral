function [ reg ] = regularisationCost(NN, lambda)
%REGULARISATIONCOST Summary of this function goes here
%   Detailed explanation goes here
     reg = 0;
     nWeightMatracies = length(NN.shape) - 1;
     for i = 1:nWeightMatracies
         Weights = reshapeWeights(NN,i);
         reg = reg + (sum(sum(Weights(2:end, :).^2)));  
     end
     
     reg = lambda/2 * reg;
     
end

