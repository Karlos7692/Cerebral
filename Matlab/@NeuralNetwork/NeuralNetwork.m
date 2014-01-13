%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% Title: Neural Network Class                                                                                  %
%                                                                                                              %
% Author: Karl Nelson                                                                                          %
% Email: <k.c.nelson7692@gmail.com>                                                                            %
%                                                                                                              %
% Description:                                                                                                 %
%                                                                                                              %
% The generic neural network structure.                                                                        %
%                                                                                                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function NN = NeuralNetwork(shape, state, nnType)
    %Constants
    NN.GENERIC = "gen";
    NN.SRN = "srn";
    NN.TEMPORAL = "tem";
    
    %Fields
    NN.shape = shape;
    NN.state = state;
    NN.weights = [];
    NN.type = nnType;
    
    for i = 2:length(shape)    
        weights = zeros(shape(i-1), shape(i));
        weights = weights(:);
        NN.weights = [NN.weights; weights];
    end
    
end 
