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
% Parameters:
%
%
%
% Fields:
%
% NN.shape    : Neural network shape, number of neurons per layer not including bias unit.
%               State units are included as input units - NN.shape(1)
% NN.state    :
% NN.weights  : All weights of the neural network in a column vector, including the bias weights 
%
%
% Constants:
% 
%
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




function NN = NeuralNetwork(shape, state, output, outenc, nnType, seed)
    %Constants
    NN.GENERIC = "gen";
    NN.SRN = "srn";
    NN.TEMPORAL = "tem";
    
    %Fields
    NN.shape = shape;
    NN.state = state;
    NN.weights = [];
    NN.type = nnType;
    NN.seed = seed;
    NN.output = output;
    NN.outenc = outenc;
    
    for i = 2:length(shape)
        
        
        %Get neural net seed.
        s = seed(i-1);
        
        %Add bias weight to previous layer shape.
        pl = shape(i-1)+1;
        
        %current layer shape has no bias
        cl = shape(i);
        
        %Randomly initialise weight matrix for i-1 - i 
        weights = randomInit(pl, cl, s);
        weights = weights(:);
        
        %Convert weight matrix to singular column vector
        NN.weights = [NN.weights; weights];
    end
    
end 





function NN = NeuralNetwork(shape, state, output, nnType, seed)
    %Constants
    NN.GENERIC = "gen";
    NN.SRN = "srn";
    NN.TEMPORAL = "tem";
    
    %Fields
    NN.shape = shape;
    NN.state = state;
    NN.weights = [];
    NN.type = nnType;
    NN.seed = seed;
    NN.output = output;
    
    for i = 2:length(shape)
        
        
        %Get neural net seed.
        s = seed(i-1);
        
        %Add bias weight to previous layer shape.
        pl = shape(i-1)+1;
        %current layer shape has no bias
        cl = shape(i);
        
        %Randomly initialise weight matrix for i-1 - i 
        weights = randomInit(pl, cl, s);
        weights = weights(:);
        
        %Convert weight matrix to singular column vector
        NN.weights = [NN.weights; weights];
    end
    
end 





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
    NN.seed = [];
    NN.output = shape(end);
    
    for i = 2:length(shape)
    
        %Get random seed.
        seed = rand("seed");
        
        %set seed for weights between layer
        NN.seed = [NN.seed,seed];
        
        %Add bias weight to previous layer shape.
        pl = shape(i-1)+1;
        %current layer shape has no bias
        cl = shape(i);
        
        %Randomly initialise weight matrix for i-1 - i 
        weights = randomInit(pl, cl, seed);
        weights = weights(:);
        
        %Convert weight matrix to singular column vector
        NN.weights = [NN.weights; weights];
    end
    
end 


function NN = NeuralNetwork(shape, state, nnType, seed)
    %Constants
    NN.GENERIC = "gen";
    NN.SRN = "srn";
    NN.TEMPORAL = "tem";
    
    %Fields
    NN.shape = shape;
    NN.state = state;
    NN.weights = [];
    NN.type = nnType;
    NN.output = shape(end);
    NN.seed = seed;
    
    for i = 2:length(shape)
        
        
        %Get neural net seed.
        s = seed(i-1);
        
        %Add bias weight to previous layer shape.
        pl = shape(i-1)+1;
        %current layer shape has no bias
        cl = shape(i);
        
        %Randomly initialise weight matrix for i-1 - i 
        weights = randomInit(pl, cl, s);
        weights = weights(:);
        
        %Convert weight matrix to singular column vector
        NN.weights = [NN.weights; weights];
    end
    
end 




function NN = NeuralNetwork(shape, state, output, nnType)
    %Constants
    NN.GENERIC = "gen";
    NN.SRN = "srn";
    NN.TEMPORAL = "tem";
    
    %Fields
    NN.shape = shape;
    NN.state = state;
    NN.weights = [];
    NN.type = nnType;
    NN.seed = [];
    NN.output = output;
    
    for i = 2:length(shape)
    
        %Get random seed.
        seed = rand("seed");
        
        %set seed for weights between layer
        NN.seed = [NN.seed,seed];
        
        %Add bias weight to previous layer shape.
        pl = shape(i-1)+1;
        %current layer shape has no bias
        cl = shape(i);
        
        %Randomly initialise weight matrix for i-1 - i 
        weights = randomInit(pl, cl, seed);
        weights = weights(:);
        
        %Convert weight matrix to singular column vector
        NN.weights = [NN.weights; weights];
    end
    
end 



function W = randomInit(nIn, nOut, seed)
    const = sqrt(6);
    epsilon_init = const/(nIn + nOut);
    rand("seed", seed);
    W = rand(nIn, nOut) * 2 * epsilon_init - epsilon_init;
end
