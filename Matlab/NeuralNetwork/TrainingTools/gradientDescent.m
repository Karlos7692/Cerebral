function [NN, E_Hist] = gradientDescent(maxIter, costType, NN, Y, X, alpha, mew, lambda)
    E_Hist = zeros(maxIter, 1);
    prevGrad = 0;
    for i = 1:maxIter
        %Neural Network maintenance: Set correct TData
        X = maintainNN(NN,X);

        
        [Weights, As, ts, Sig_grads, Hx, reg ] = forwardPass(NN, X, lambda);            
        [E, gradconst] = costFunction(Y, Hx, reg, costType);
        [ grad ] = backwardPass(NN, Y, Hx, Weights, As, Sig_grads, ts, gradconst);
        
        
        %Add momentum
        mom = mew * prevGrad;
        
        
        %update weights
        NN.weights = NN.weights - alpha*grad + mom;
        
        %update momentum
        prevGrad = alpha*grad;
        
        E_Hist(i) = E;
        
    end
end

