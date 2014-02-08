function works = gradientDescentChecking(maxIter, NN, Y, X, alpha, mew, lambda)
J_Hist = zeros(maxIter, 1);
    prevGrad = 0;
    i = 1;
    works = 1;
    while ((i < maxIter) & works)
        %Neural Network maintenance: Set correct TData
        
        X = maintainNN(NN,X);
        [J, grad] = costFunc(NN, Y, X, lambda);
        
        diff = checkGradient(NN, Y, X, lambda);
        if(diff > 1*e-9)
            works = 0;
        else
        
        %Add momentum
        grad = mew * prevGrad + (1 - mew) * grad;
        
        %update weights
        NN.weights = NN.weights +  alpha * grad;
        
        %update momentum
        prevGrad = grad;
        
        %update iterator.
        i = i + 1;
        end
        
    end



end
