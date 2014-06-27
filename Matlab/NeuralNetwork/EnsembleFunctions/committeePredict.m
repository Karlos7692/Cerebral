function [ Out ] = committeePredict( Committee, X, weights)
%COMMITTEEPREDICT Summary of this function goes here
%   Detailed explanation goes here
    nNNs = length(Committee);
    %TODO Put in general form
    Result = zeros(size(X,1), nNNs);
    for i = 1:nNNs
        Result(:,i) = weights(i) * (predict(Committee{i}, X)); 
    end
    
    Out = sum(Result,2);
end

