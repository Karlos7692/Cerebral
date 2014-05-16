function [ grad, delta_k ] = bakwardPassError(delta_k, NN, Weights, As, Sig_grads, ts, gc)
%BAKWARDPASSERROR Summary of this function goes here
%   Detailed explanation goes here
    grad = zeros(size(NN.weights));                                 %Preallocate grad vector
    lastWeight = numel(NN.weights);                                 %Last weight position for current layer
    
    % Already contains the Sig_grads of each set of Output Neurons.
    
    for i = length(Weights):-1:2                                    %Go backwards from layer n-1:n to 1:2 
        grad_Wi = gc * (As{i}' * delta_k + ts{i});
        firstWeight = lastWeight - numel(grad_Wi) + 1;              %Get first weight position
        grad(firstWeight:lastWeight) = grad_Wi(:);                  %Update Gradient for l-1:l
        lastWeight = lastWeight - numel(grad_Wi);                   %Update Last  Weight Position
        delta_k = delta_k * Weights{i}(2:end,:)' .* Sig_grads{i-1}; %Update Error Signal
    end
    %Update first layer - second layer weights
    grad_W1to2 = gc *(As{1}' * delta_k + ts{1});
    grad(1:lastWeight) = grad_W1to2(:);

end

