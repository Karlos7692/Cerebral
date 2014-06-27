function [ CNN ] = generateCNN(feats, nns, tier1, tier2)
%GENERATECNN Summary of this function goes here
%   Detailed explanation goes here
     
     %rows and cols of component matrix,
     
     %Initial Neural Network. Zero Tier.
     [ CNN ] = buildConvolutionalNetwork();
     
     %1st Tier Neural Network
     [ ComponentMatrix ] = generateComponentMatrix(feats, nns);
     [ Component ] = buildComponent(tier1, 1, 1, ComponentMatrix);
     [ CNN ] = appendComponent(CNN, Component, ComponentMatrix);
     
     
     %2nd Tier Neural Network
     [ ComponentMatrix ] = generateComponentMatrix(nns, 1);
     [ Component ] = buildComponent(tier2, 1, 1, ComponentMatrix);
     [ CNN ] = appendComponent(CNN, Component, ComponentMatrix);
     
     %Final Data Transformation after CNN: Output
     [ CNN ] = appendOutputMatrix(CNN, 1);
     
end

