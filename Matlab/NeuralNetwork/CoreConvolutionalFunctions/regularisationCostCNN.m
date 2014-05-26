function [ reg ] = regularisationCostCNN(CNN, lambda)
%REGULARISATIONCOSTCNN Summary of this function goes here
%   Detailed explanation goes here
    reg = 0;
    for com = 1:length(CNN.Components)
        for cel = 1:length(CNN.Components{com})
            [ regTemp ] = regularisationCost(CNN.Components{com}{cel}, lambda);
            reg = reg + regTemp;
        end
    end

end

