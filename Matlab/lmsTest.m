function [J, grad] = lmsTest(nn_params, insize, hiddensize, outsize, X, Y, lambda)
%TODO Generalise for shape.    
    
    

    m = size(X,1);
   
    
    W1 = reshape(nn_params(1:hiddensize * (insize + 1)), ...
                 (insize+1), hiddensize);

    W2 = reshape(nn_params((1 + (hiddensize * (insize + 1))):end), ...
                 (hiddensize+1), outsize);

    
    A1 = [ones(m,1), X];
    Z2 = sigmoid(A1*W1);
    A2 = [ones(m,1), Z2];
    Hx = sigmoid(A2*W2);

    J =   1/2 * sum(sum((Y - Hx).^2)) + lambda/2 * (sum(sum(W1(2:end, :).^2)) + sum(sum(W2(2:end, :).^2)));
    

    %Backprop gradient
    %Regualrisation terms
    t1 = [zeros(1, size(W1,2)); (lambda * W1(2:end, :))];
    t2 = [zeros(1, size(W2,2)); (lambda * W2(2:end, :))];
    
    %delta_k = dE/dOut 
    delta_k = (-1) * (Y - Hx) .* Hx .* (ones(size(Hx)) - Hx);
    grad_W2 = (A2' * delta_k + t2);
    
    %delta_k = dE/dHidden
    delta_k = delta_k * W2(2:end, :)' .* (Z2 .* (ones(size(Z2)) - Z2));
    grad_W1 =  (A1' * delta_k + t1);
    
    grad = [grad_W1(:) ; grad_W2(:)];
end

