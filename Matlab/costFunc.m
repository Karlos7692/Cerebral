function J = costFunc(NN, Y, X, lambda)



end



function ce = crossEntropyCost(NN, Y, X, lambda)
    A = X;
    for i = 1:(length(NN.shape)-1)
        W = reshapeWeights(NN,i);
        Z = [ones(size(A,1)), A];
        A = sigmoid(Z*W);
    end
    Hx = A;
    ce =  1/m * sum(-Y.*log(Hx) - (ones(size(Y))-Y).*log(ones(size(Y))-Hx));

    %Find Regularisation cost
    %Add regularisation cost to cross entropy cost.
end
