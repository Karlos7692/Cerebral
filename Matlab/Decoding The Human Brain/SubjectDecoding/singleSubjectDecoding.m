function [ NN, kfacc, avgtacc, KF_E_Hist, E_Hist ] = singleSubjectDecoding( k, X, Y, NN, traintype, maxIter, batchsize, alpha, mew, lambda)
%SINGLESUBJECTDECODING Summary of this function goes here
%   Detailed explanation goes here
    
    %Run K-Fold Evluation on NN. Uses Stub NN to get accuracy
    fprintf('Running K-Fold Evaluation...\n');
    [ kfacc, avgtacc, KF_E_Hist ] = kfoldValidation(k, X, Y, NN, traintype, maxIter, batchsize, alpha, mew, lambda);
    fprintf('Now Training Neural Network...\n');
    [NN, E_Hist] = train(X, Y, NN, traintype, maxIter, batchsize, alpha, mew, lambda);
    
end

