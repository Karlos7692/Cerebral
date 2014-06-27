function [ distWeights, avgEntropy ] = learnDistributionWeights(YTrain, XTrain, YTest, XTest)
%LEARNDISTRIBUTIONWEIGHTS Summary of this function goes here
%   Detailed explanation goes here
    
    totalEntropy = 0;
    defaultCorr = 0.99;
    distWeights = zeros(size(YTrain));
    for i = 1:size(XTrain,1)

     xtrain = XTrain(i,:);
     class = YTrain(i,:);
     
     %Get probability pos match. Wave corresponds to coreect class. 
     XPosTest = XTest((YTest == class),:);      
     [ coverPosTest ] = calculateCover(xtrain, XPosTest, defaultCorr);
     XPosTrain = XTrain(YTrain == class,:);
     [ coverPosTrain ] = calculateCover(xtrain, XPosTrain, defaultCorr);
     %Must remove the positive result for comparing xtrain with xtrain
     coverPosTrain = (coverPosTrain * size(XPosTrain,1) - 1)/size(XPosTrain,1)-1;
     pPos = (coverPosTest + coverPosTrain)/(size(XTest,1) + size(XTrain,1));
     
     %Get probability neg match. Wave actually corresponds to
     %opposite class
     XNegTest = XTest((YTest ~= class),:);      
     [ coverNegTest ] = calculateCover(xtrain, XNegTest, defaultCorr);
     XNegTrain = XTrain(YTrain ~= class,:);
     [ coverNegTrain ] = calculateCover(xtrain, XNegTrain, defaultCorr);
     pNeg = (coverNegTest + coverNegTrain)/(size(XTest,1) + size(XTrain,1));
     
     %Calculate entropy
     pvec = [pPos, pNeg];
     [ entropy ] = calculateEntropy( pvec )
     totalEntropy = totalEntropy + entropy;
     
     %Should be weighted according to information gain.
     %High entropy indicates that the wave does not tell say much about the
     %class. log2 to bound large weights existing in test set.
     distWeights(i) = (pPos - pNeg)/(size(XTest,1) + size(XTrain,1));
    end
    
    %Average Entropy given a subject.
    avgEntropy = totalEntropy/size(XTrain,1);
end

