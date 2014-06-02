






function [NN, J_Hist] = gradientDescent(maxIter, NN, Y, X, alpha, mew, lambda)
    J_Hist = zeros(maxIter, 1);
    prevGrad = 0;
    for i = 1:maxIter
        %Neural Network maintenance: Set correct TData
        X = maintainNN(NN,X);

        
        [J, grad] = costFunc(NN, Y, X, lambda);
        
        
        %Add momentum
        mom = mew * prevGrad;
        
        
        %update weights
        NN.weights = NN.weights - alpha*grad + mom;
        
        %update momentum
        %prevGrad = grad;
        J_Hist(i) = J;
        
    end
end
