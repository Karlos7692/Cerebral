function [ CNN, TrainData, TargDatas ] = buildConvolutionalNetwork(RawIn, RawOuts, NNParamsList, ConvolutionalParams)
%BUILDCONVOLUTIONALNETWORK Summary of this function goes here
%   Detailed explanation goes here
    structure = ConvolutionalParams.structure;
    nComponents = length(structure);
    ComponentConnections = ConvolutionalParams.ComponentConnections;
    %Transform RawIn into correct format to meet requirements for Component 1:
    %TODO Generalise
    encoding = NNParamsList{1}.encoding;                       
    inenc = encoding(1);
   
    %%TODO Build Training Data
    TrainData = RawIn;
    
    %Building Target Data:
    TargDatas = cell(size(RawOuts));
    
    %Building Components:
    Components = cell(1, length(nComponents));
    
    %Build First Component: %TODO Generalise for temporal 1st component.
    %TODO: Change ComponentConnection{1} Matrix.
    %TODO: Change RawOut to general NN propblem
    %Convert to starting TrainData from RawIn
    RawInSplits = transform(ComponentConnections{1}, RawIn, 1);
    Component = cell(structure(1), 1);
    for i = 1:structure(1)
        NN = buildNeuralNetworkFromData(RawInSplits{i}, RawOuts{1}, NNParamsList{1}.hidden,...
            NNParamsList{1}.nStateVecs, NNParamsList{1}.type, NNParamsList{1}.encoding);
        Component{i} = NN;
    end
    Components{1} = Component;
    TargDatas{1} = convertRawOut(RawOuts{1}, NNParamsList{1});
    
    %TODO Change input values to proper encoding
    %TODO Change ComponentConnections to match Changed Input Encoding 
    

    
    %Bulild Components - All Components are Cells
    for i = 2:nComponents
       NNParams = NNParamsList{i}; 
       [TargDataTemp, nOuts, output] = convertRawOut(RawOut, NNParams);
       Component = cell(structure(i), 1);
       for j = 1:structure(i)
           nIns = size(ComponentConnections{i}, 2);   
           NN = buildNeuralNetwork(nIns, hidden, nOuts, output, NNParams.encoding(2));
           Component{j} = NN;
       end
       Components{i} = Component;
       TargDatas{i} = TargDataTemp;
    end
    
    
    
    
    %Build Convolutional Network
    CNN = ConvolutionalNeuralNetwork(Components, ComponentConnections, stucture, multiStage);

end



function [TargData, nOuts, output] = convertRawOut(RawOut, NNParams)
    outenc = NNParams.encoding(2);
    TargData = [];
    output = [];
    for i = 1:size(RawOut,2);
        if (outenc == BINARY)
            TargTemp = convertData(RawOut(:, i), BINARY);
        elseif (outenc == BINARYDEC)
            TargTemp = convertData(RawOut(:, i), BINARYDEC);
        else
            TargTemp = RawOut(:,i);
        end
        TargData = [TargData, TargTemp];
        output = [output, size(TargTemp, 2)];
    end
   nOuts = length(TargData);
end

    
  

