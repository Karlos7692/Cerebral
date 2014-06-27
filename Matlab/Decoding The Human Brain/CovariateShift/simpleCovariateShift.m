function [ DistWeights ] = simpleCovariateShift( Dists, Outs, NNs, alphas, mews, lambdas)
%SIMPLECOVARIATESHIFT Summary of this function goes here
%   Detailed explanation goes here

    DistWeights = cell(size(Dists));
    for i = 1:length(Dists)
        training = floor(8/10 * size(Dists{i}, 1));
        XTrain = Dists{i}(1:training,:);
        YTrain = Outs{i}(1:training,:);
        XCV = Dists{i}(training+1:end,:);
        YCV = Outs{i}(training+1:end,:);
        tic;
        [NN] = gradientDescent(100, 'lms', NNs{i}, YTrain, XTrain, alphas(i), mews(i),lambdas(i));
        toc;
        [XTest, YTest] = concatDists(Dists, Outs, i);
        Hx = predict(NN, XTest) >= 0.5;
        PTest = sum(Hx == YTest)/size(YTest,1);
        Hx = predict(NN, XCV) >= 0.5;
        PTrain = sum(Hx == YCV)/size(YCV,1);
        
        %PTest/PTrain gives probability of test given all xs of subject.
        DistWeights{i} = ones(size(Dists{i},1),1) * (PTest/PTrain); 
                                                               
    end

end
