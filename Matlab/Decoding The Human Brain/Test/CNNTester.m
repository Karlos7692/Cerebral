function [RawIn, Y, CNN, mu, scale] = CNNTester(X, y, nNNs, nTrials, nIns)
%CNNTESTER Summary of this function goes here
%   Detailed explanation goes here
   RawInTemp = X(1:nTrials ,1:nNNs ,end-nIns+1:end);
   Y= double(y(1:nTrials));
   [RawIn, mu, scale] = prepareRawData(RawInTemp);
   [ CNN ] = generateCNN(RawInTemp);
end

