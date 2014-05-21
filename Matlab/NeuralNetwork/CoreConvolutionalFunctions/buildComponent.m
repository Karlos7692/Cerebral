function [ Component ] = buildComponent(hidden, nOuts, outvec, ComponentMatrix)
%BUILDCOMPONENT Summary of this function goes here
%   Detailed explanation goes here

    %TODO change Component Matrix to produce uneven sized NNs.
    nNNs = size(ComponentMatrix,2);
    Component = cell(1,nNNs);
    for i = 1:nNNs
        nIns = size(ComponentMatrix, 1);
        
        %NN will always have a binary encoding in a component.
        %Components are then morphed into correct type, and encodings
        %in problem level. Since the type and econding is problem specific.
        %Outenc = 1 refers to a binary encoding.
        NN = buildNeuralNetwork(nIns, hidden, nOuts, outvec, 1);
        Component{i} = NN;
    end
end

