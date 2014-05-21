function [ NN ] = buildNeuralNetwork(nIns, hidden, nOuts, output, outenc)
%BUILDNEURALNETWORK Summary of this function goes here
%   Detailed explanation goes here

      GENERAL = 'gen';     
      shape = [nIns, hidden, nOuts];
      seedvec = generateSeedvec(shape);
      NN = NeuralNetwork(shape, [],  output, outenc, GENERAL, seedvec);
end

