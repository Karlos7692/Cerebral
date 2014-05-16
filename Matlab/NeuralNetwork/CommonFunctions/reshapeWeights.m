%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% Title: Reshape Neural Network Layer                                                                          %
%                                                                                                              %
% Author: Karl Nelson                                                                                          %
% Email: <k.c.nelson7692@gmail.com>                                                                            %
%                                                                                                              %
% Description:                                                                                                 %
%                                                                                                              %
% Reshapes a specific neural network layer                                                                     %
%                                                                                                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function Weights = reshapeWeights(NN, layerFrom)
       
    shape = NN.shape;
    if ((layerFrom >= 1) && (layerFrom < length(shape)))
        
        
        
        start = 0;
        for i = 2:layerFrom
            %Include bias term for each predecessor layer.
            start = start + (shape(i-1)+1) * shape(i);
        end
    
        start = start + 1; 
        layerTo = layerFrom + 1;
      
        %Include bias term for in predecessor layer.
        nWeights = (shape(layerFrom)+1) * shape(layerTo);
        finish = start + nWeights - 1;
    
        weights = NN.weights;
        weights = weights(start:finish);
        
        %Add bias neuron to from layer. No bias in layerTo
        Weights = reshape(weights, (shape(layerFrom)+1), shape(layerTo));
        
        %Bias Weights are the last row vector.
    end
end

