function [ numgrad ] = approxGradient(NN, TrainData, TargData, lambda)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    numgrad = zeros(size(NN.weights));
    perturb = zeros(size(NN.weights));
    e = 1e-4;
    for p = 1:numel(NN.weights)
    % Set perturbation vector
    perturb(p) = e;
    NNTestLow = NN;
    NNTestUp = NN;
    NNTestLow.weights = NN.weights - perturb;
    NNTestUp.weights = NN.weights + perturb;
    loss1 = lmsCost(NNTestLow, TrainData, TargData, lambda);
    loss2 = lmsCost(NNTestUp, TrainData, TargData, lambda);
    % Compute Numerical Gradient
    numgrad(p) = (loss1 - loss2) / (2*e);
    perturb(p) = 0;
    end
end

function J = lmsCost(NNTest, TrainData, TargData, lambda)
    m = size(TrainData,1); 
    Hx = predict(NNTest, TrainData);
    J =   1/(2*m) * sum(sum((TargData - Hx).^2));
end
