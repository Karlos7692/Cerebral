function [ NN, E_Hist] = stocGradientDescent(maxIter, batchSize, costType, NN, Y, X, alpha, mew, lambda)
%STOCGRADIENTDESCENT Summary of this function goes here
%   Detailed explanation goes here

    nTraining = size(X,1);
    nBatches = floor(nTraining/batchSize);
    E_Hist = zeros(maxIter*nBatches, 1);
    prevGrad = 0;
    
    k = 1;
    %Remainder to redistribute for uneven batch size    
    rem = mod(nTraining, batchSize);    
    for i = 1:maxIter
        
        %Neural Network maintenance: Set correct TData
        X = maintainNN(NN,X);
        
        %Extra examples to redistribute
        extra = rem;
        fst = 1;
        lst = 0;
        for j = 1:nBatches
            
            
            if extra > 0
                %Extra Training examples still left to distribute.
                lst = lst + batchSize + floor(rem/nBatches);
                extra = extra - floor(rem/nBatches);
                
                if j == nBatches
                    lst = lst + extra;
                    extra = 0;
                end
            else
                %No extra training examples to redistribute
                lst = lst + batchSize;
            end
              
            [Weights, As, ts, Sig_grads, Hx, reg ] = forwardPass(NN, X(fst:lst,:), lambda);            
            [E, gradconst] = costFunction(Y(fst:lst,:), Hx, reg, costType);
            [ grad ] = backwardPass(NN, Y(fst:lst,:), Hx, Weights, As, Sig_grads, ts, gradconst);
            fst = lst + 1;
           
            %Add momentum
            mom = mew * prevGrad;
        
        
            %update weights
            NN.weights = NN.weights - alpha*grad - mom;
        
            %update momentum
            prevGrad = alpha*grad;
        
            E_Hist(k) = E;
            k = k + 1;
        end
    end

end

