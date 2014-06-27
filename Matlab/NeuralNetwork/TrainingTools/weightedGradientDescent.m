function [ output_args ] = weightedGradientDescent( maxIter, DistWeights, costType, NN, Y, X, alpha, mew, lambda )
%WEIGHTEDGRADIENTDESCENT Summary of this function goes here
%   Detailed explanation goes here
    E_Hist = zeros(maxIter, 1);
    prevGrad = 0;
    for i = 1:maxIter
        %Neural Network maintenance: Set correct TData
        X = maintainNN(NN,X);

        
        [Weights, As, ts, Sig_grads, Hx, reg ] = forwardPass(NN, X, lambda);            
        [E, gradconst] = costFunction(Y, Hx, reg, costType);
        [ grad ] = weightedBackwardPass( NN, DistWeights, Y, Hx, Weights, As, Sig_grads, ts, gradconst );
        
        
        %Add momentum
        mom = mew * prevGrad;
        
        
        %update weights
        NN.weights = NN.weights - alpha*grad - mom;
        
        %update momentum
        prevGrad = alpha*grad;
        
        E_Hist(i) = E;
        
    end

end

