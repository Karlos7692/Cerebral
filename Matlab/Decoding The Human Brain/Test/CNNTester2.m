function [RawIn, Y, CNN, mu, scale] = CNNTester2(X, y, nNNs, nTrials, nIns)
%CNNTESTER2 Summary of this function goes here
%   Detailed explanation goes here
   RawInTemp = X(1:nTrials ,1:nNNs ,end-nIns+1:end);
   Y= double(y(1:nTrials));
   [RawIn, mu, scale] = prepareRawData(RawInTemp);
   
    %rows and cols of component matrix,
     rows = size(RawInTemp,3); cols = size(RawInTemp,2);
     
     %Initial Neural Network. Zero Tier.
     [ CNN ] = buildConvolutionalNetwork();
     
     %1st Tier Neural Network
     [ ComponentMatrix ] = generateComponentMatrix(rows, cols);
     [ Component ] = buildComponent(30, 1, 1, ComponentMatrix);
     [ CNN ] = appendComponent(CNN, Component, ComponentMatrix);
     
     
     [ CNN ] = appendOutputMatrix(CNN, 1);
end

