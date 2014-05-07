function [ numgrad ] = adjustApproxGradient(NN, TrainData, TargData, lambda, acc)
%ADJUSTAPPROXGRADIENT Summary of this function goes here
%   Detailed explanation goes here
    numgrad = zeros(size(NN.weights));
    perturb = zeros(size(NN.weights));
    e = 1 * 10^(-acc);
    for p = 1:numel(NN.weights)
    % Set perturbation vector
    perturb(p) = e;
    NNTestLow = NN;
    NNTestUp = NN;
    NNTestLow.weights = NN.weights - perturb;
    NNTestUp.weights = NN.weights + perturb;
    loss1 = costFunc(NNTestLow, TargData, TrainData, lambda);
    loss2 = costFunc(NNTestUp, TargData, TrainData, lambda);
    % Compute Numerical Gradient
    numgrad(p) = (loss2 - loss1) / (2*e);
    perturb(p) = 0;
    end

end

