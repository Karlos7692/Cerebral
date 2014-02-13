function [J grad] = lmsCost( nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%LMSCOST Summary of this function goes here
%   Detailed explanation goes here
    

    W1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

    W2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
             
    m = size(X,1);
    for i = 1:m
        Y(i,y(i)) = 1;
    end
    
    A1 = [ones(m,1), X];
    Z2 = sigmoid(A1*W1');
    A2 = [ones(m,1), Z2];
    Hx = sigmoid(A2*W2');

    J =   1/2 * sum(sum((Y - Hx).^2)) + lambda/2 * (sum(sum(W1(:, 2:end).^2)) + sum(sum(W2(:, 2:end).^2)));
    

    %Backprop gradient
    %Regualrisation terms
    t1 = [zeros(size(W1,1),1), (lambda * W1(:, 2:end))];
    t2 = [zeros(size(W2,1),1), (lambda * W2(:, 2:end))];
    
    %delta_k = dE/dOut 
    delta_k = (-1) * (Y - Hx) .* Hx .* (ones(size(Hx)) - Hx);
    grad_W2 = (delta_k'*A2 + t2);
    
    %delta_k = dE/dHidden
    delta_k = delta_k * W2(:, 2:end) .* (Z2 .* (ones(size(Z2)) - Z2));
    grad_W1 =  (delta_k'*A1 + t1);
    
    grad = [grad_W1(:) ; grad_W2(:)];

end


