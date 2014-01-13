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
            start = start + shape(i-1) * shape(i);
        end
    
        start = start + 1; 
        layerTo = layerFrom + 1;
        nWeights = shape(layerFrom) * shape(layerTo)
        finish = start + nWeights - 1;
    
        weights = NN.weights;
        weights = weights(start:finish);
        Weights = reshape(weights, shape(layerFrom), shape(layerTo));
    end
end

