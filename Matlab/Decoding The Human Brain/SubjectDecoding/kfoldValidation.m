function [ kfacc, avgtacc, E_History ] = kfoldValidation(k, X, Y, NN, traintype, maxIter, batchsize, alpha, mew, lambda)
%KFOLDVALIDATION Summary of this function goes here
%   Detailed explanation goes here
    tic;
    fprintf('%d-fold CV || ',k);
    m = size(X,1); 
    cvsize = floor(m/k);
    avgtacc =0;
    kfacc = 0;
    fst = 1;
    E_History = zeros(maxIter,k);

    for i = 1:k
        if i == k
            lst = m; %May not be distributed properly.
        else
            lst = i*cvsize;
        end
        NNTemp = NN;
        cvset = fst:lst;
        %Setup Training Set, Everything but the cvset
        trainset = ones(m,1);
        trainset(fst:lst) =  0;
        trainset = logical(trainset);
        
        XTrain = X(trainset,:); 
        YTrain = Y(trainset,:);
        
        
        
        [NNTemp, E_Hist] = train(XTrain, YTrain, NNTemp, traintype, maxIter, batchsize, alpha, mew, lambda);
        
        %Predictions:
        %Predict on tset:
        TOut = predict(NNTemp, XTrain) >= 0.5;
        %Predict: on cvset
        CVOut = predict(NNTemp, X(cvset,:)) >= 0.5;
        
        %Get Accuracy:
        tacc = sum(TOut == logical(YTrain))/size(YTrain,1);
        cvacc = sum(CVOut == logical(Y(cvset,:)))/length(cvset);
        
        %Get Results
        avgtacc = avgtacc + tacc;
        kfacc = kfacc + cvacc;
        fst = lst +1;

        E_History(:,i) = E_Hist;
        
    end  
    avgerr = mean(E_History(end,:));
    kfacc = kfacc/k;
    avgtacc = avgtacc/k;
    
    %Print Results
    fprintf(' Average Error: %f  ||  Avg Training Acc: %f K-Fold Acc: %f  || ', avgerr, avgtacc, kfacc);
    toc;

end

