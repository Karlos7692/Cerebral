






function [NN, J_Hist] = gradientDescent(maxIter, NN, Y, X, alpha, mew, lambda)
    J_Hist = zeros(maxIter, 1);
    prevGrad = 0;
    for i = 1:maxIter
       
        [J, grad] = costFunc(NN, Y, X, lambda);
        
        %Add momentum
        grad = mew * prevGrad + (1 - mew) * grad;
        
        %update weights
        NN.weights = NN.weights +  alpha * grad;
        
        %update momentum
        prevGrad = grad;
        J_Hist(i) = J;
        
    end
end
