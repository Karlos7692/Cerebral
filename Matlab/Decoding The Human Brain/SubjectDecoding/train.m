
function [NN, E_Hist] = train(X, Y, NN, traintype, maxIter, batchsize, alpha, mew, lambda)

    E_Hist = [];
    if strcmp(traintype, 'stoc')
        [NN, E_Hist] = gradientDescent(maxIter, 'lms', NN, Y,X,alpha, mew, lambda);
        
    elseif strcmp(traintype, 'grad')
        [NN, E_Hist] = gradientDescent(maxIter,'lms',NN,Y,X,alpha,mew,lambda);
    end   
end

