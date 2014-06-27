function [ bestlambda, bestkfacc ] = learnBestNNParams(X, Y)
%LEARNBESTNNPARAMS Summary of this function goes here
%   Detailed explanation goes here

    alpha = 0.001;
    mew = 0.0003;
    k=6;
    lambdas = 6:1:20;
    [NN] = buildNeuralNetworkFromData(X, Y, [40,10], 0, 'gen', [5,1]);
    
    kfPlot = zeros(1,length(lambdas));
    taccPlot = zeros(1,length(lambdas));
    
    %Permute Training data
    perm = randperm(size(X,1));
    X = X(perm,:);
    Y = Y(perm,:);
    
    bestkfacc = 0;
    bestlambda = 0;
    i = 1;
    for lambda = lambdas
        [ kfacc, avgtacc, E_History ] = kfoldValidation(k, X, Y, NN, 'grad', 100, floor(size(X,1)/k), alpha, mew, lambda);
        kfPlot(i) = kfacc;
        taccPlot(i) = avgtacc;
        if bestkfacc < kfacc
            bestkfacc = kfacc;
            bestlambda = lambda;
        end
            
        figure;
        plot(E_History);
        i = i+1;
    end
    
    figure;
    plot(taccPlot, 'b');
    hold on;
    plot(kfPlot,'r');
    hold off;

end

