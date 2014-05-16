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
    HxL = predict(NNTestLow, TrainData);
    HxU = predict(NNTestUp, TrainData);
    %TODO Get regularisation:
    regL =  regularisationCost(NNTestLow, lambda);
    regU =  regularisationCost(NNTestUp, lambda);
    
    loss1 = costFunction(TargData, HxL, regL, 'lms');
    loss2 = costFunction(TargData, HxU, regU, 'lms');
    
    % Compute Numerical Gradient
    numgrad(p) = (loss2 - loss1) / (2*e);
    perturb(p) = 0;
    end
end
