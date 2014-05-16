function [ WeightsCell, AsCell, tsCell, Sig_gradsCell, HxCell, regCell ] = forwardPassComponent(CNN, X, lambda, index)
%EVALUATECOMPONENTS Summary of this function goes here
%   Detailed explanation goes here

    WeightsCell = cell(length(CNN.Component{index}), 1);
    AsCell = cell(length(CNN.Component{index}), 1);
    tsCell = cell(length(CNN.Component{index}), 1);
    Sig_gradsCell = cell(length(CNN.Component{index}), 1);
    HxCell = cell(length(CNN.Component{index}), 1);
    regCell = cell(length(CNN.Component{index}), 1);
    
    XSplits = transform(CNN.ComponentConnections{index}, X, index);
    for j = 1:length(CNN.Component{index})
        
        [Weights, As, ts, Sig_grads, Hx, reg ] = forwardPass(CNN.Components{index}{j}, XSplits{j}, lambda);
        WeightsCell{j} = Weights;
        AsCell{j} = As;
        tsCell{j} = ts;
        Sig_gradsCell{j} = Sig_grads;
        HxCell{j} = Hx;
        regCell{j} = reg;
    end 
    
    

end

