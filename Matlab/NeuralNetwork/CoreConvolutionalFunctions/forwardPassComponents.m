function [ WeightsCollection, AsCollection, tsCollection, Sig_gradsCollection, HxCell, reg] ...
     = forwardPassComponents(CNN, XCell, lambda)
%FORWARDPASSCOMPONENTS Summary of this function goes here
%   Detailed explanation goes here
   
    %For all Components Forward Pass till the Final Component.
    WeightsCollection = cell(length(CNN.structure)-1, 1);
    AsCollection = cell(length(CNN.structure)-1, 1);
    tsCollection = cell(length(CNN.structure)-1, 1);
    Sig_gradsCollection = cell(length(CNN.structure)-1, 1);
    
    %X is the cell pre-transformed from last Component. The Original input 
    %is of type cell.
    
    %X_t+1 and regCell is used for calculating Convolutional Cost, thus we do
    %not need to collect them. We only return the value.
    %HxCell = X_end
    reg = 0;
    for i=1:length(CNN.structure)
       [ WeightsCell, AsCell, tsCell, Sig_gradsCell, XCell, regCell ] = forwardPassComponent(CNN, XCell, lambda, i);
       WeightsCollection{i} = WeightsCell;
       AsCollection{i} = AsCell;
       tsCollection{i} = tsCell;
       Sig_gradsCollection{i} = Sig_gradsCell;
       
       reg = addRegCell(reg, regCell);
       
    end
    
    HxCell = XCell;
    

end


function [reg] = addRegCell(reg, regCell)
     
   for i = 1:size(regCell,1)
       reg = reg + regCell{i};
   end

end
