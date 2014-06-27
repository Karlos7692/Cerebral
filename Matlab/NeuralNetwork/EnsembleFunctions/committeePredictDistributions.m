function [ Out ] = committeePredictDistributions( Committee, Xs, weights)
%COMMITTEEPREDICTDISTRIBUTIONS Summary of this function goes here
%   Detailed explanation goes here
    nNNs = length(Committee);
    Results = zeros(size(Xs{1},1), nNNs);
    for i = 1:nNNs
        %Weighted prediction
        Results(:,i) = (predict(Committee{i}, Xs{i}) >= 0.5) * weights(i);
    end
    Out = sum(Results,2);

end

