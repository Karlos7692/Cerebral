function works = gradientDescentChecking(maxIter, NN, Y, X, alpha, mew, lambda)
    prevGrad = 0;
    i = 1;
    works = 1;
    while ((i < maxIter) & works)
        %Neural Network maintenance: Set correct TData
        
        X = maintainNN(NN,X);

        
        [diff, J, grad] = checkGradient(NN, Y, X, lambda);
        if(diff > 1e-8)
            works = 0;
        else
        
        %Add momentum
        grad = mew * prevGrad + (1 - mew) * grad;
        
        %update weights
        NN.weights = NN.weights -  alpha * grad;
        
        %update momentum
        prevGrad = grad;
        
        %update iterator.
        i = i + 1;
        end
        
    end



end
