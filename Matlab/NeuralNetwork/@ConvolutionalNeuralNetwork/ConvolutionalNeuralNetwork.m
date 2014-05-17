function [ CNN ] = ConvolutionalNeuralNetwork(Components, ComponentConnections, InvertedComponentConnections, structure, multiStage)
%CONVOLUTIONALNEURALNETWORK Summary of this function goes here
%   Detailed explanation goes here

CNN.multistageTraining = multiStage
CNN.Components = Components;
CNN.structure = structure;
CNN.ComponentConnections = ComponentConnections;
CNN.InvertedComponentConnections = InvertedComponentConnections;

end

