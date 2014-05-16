function [Weights, As, ts, Sig_grads, Hx, reg ] = forwardPass(NN, X, lambda)
%FORWARDPASS Summary of this function goes here
%   Detailed explanation goes here
    m = size(X,1);
    nWeightMatracies = length(NN.shape) - 1;
    nLayers = length(NN.shape);
    %Forward Pass
    %Get Weights Matrix Array
    %Get As, Zs and Hx
    %Get Reglarisation cost
    %Get Regualrisation terms
    %Collect sigmoid gradients
    Weights = cell(1,nWeightMatracies);     %nWs
    Zs = cell(1,nLayers);                   %nZs = nWs
    As = cell(1,nWeightMatracies);          %nAs = nWs
    ts = cell(1,nWeightMatracies);          %nts = nWs
    Zs{1} = X;
    reg = 0;
    Sig_grads = cell(1,nWeightMatracies);
    for i = 1:nWeightMatracies
        Weights{i} = reshapeWeights(NN,i);                                       %Get Weights
        As{i} = [ones(m,1), Zs{i}];                                              %Get As
        Zs{i+1} = sigmoid(As{i}*Weights{i});                                     %Get Zs
        reg = reg + (sum(sum(Weights{i}(2:end, :).^2)));                         %Get regularisation cost
        ts{i} = [zeros(1, size(Weights{i},2)); (lambda * Weights{i}(2:end, :))]; %Get regularisation terms
        Sig_grads{i} = (Zs{i+1} .* (ones(size(Zs{i+1})) - Zs{i+1}));             %Collect sigmoid gradients (starting layer 2)
    end
    reg = lambda/2 * reg;
    outLayer = length(NN.shape);
    Hx = Zs{outLayer};

end

