function [ CNN ] = generateCNN(X)
%GENERATECNN Summary of this function goes here
%   Detailed explanation goes here
     
     %rows and cols of component matrix,
     rows = size(X,3); cols = size(X,2);
     
     %Initial Neural Network. Zero Tier.
     [ CNN ] = buildConvolutionalNetwork();
     
     %1st Tier Neural Network
     [ ComponentMatrix ] = generateComponentMatrix(rows, cols);
     [ Component ] = buildComponent(30, 1, 1, ComponentMatrix);
     [ CNN ] = appendComponent(CNN, Component, ComponentMatrix);
     
     
     %2nd Tier Neural Network
     [ ComponentMatrix ] = generateComponentMatrix(cols, 1);
     [ Component ] = buildComponent(100, 1, 1, ComponentMatrix);
     [ CNN ] = appendComponent(CNN, Component, ComponentMatrix);
     
     %Final Data Transformation after CNN: Output
     [ CNN ] = appendOutputMatrix(CNN, 1);
     
end

