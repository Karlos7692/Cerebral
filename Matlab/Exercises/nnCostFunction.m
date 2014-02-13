function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%




X = [ones(m,1), X];
H1 = sigmoid(X*Theta1');
H1 = [ones(m,1), H1];
Out = sigmoid(H1*Theta2');
K = size(Out,2);
Y = zeros(m,K);

for i = 1:m
    Y(i,y(i)) = 1;
end



J = 1/m* sum(sum(-Y.*log(Out) - (ones(m,K)-Y).*log(ones(m,K)-Out))) + lambda/(2*m) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));

%D2 = Theta2_grad;
%D1 = Theta1_grad;
%for t = 1:m
    
    %Forward Pass
%    a1 = X(t,:);
%    z2 = sigmoid(a1*Theta1');
%    a2 = [1, z2];
%    z3 = sigmoid(a2*Theta2');
%    a3 = z3;
    
    %a3(1,:)'
    %Cost = 1/m* sum(sum(-Y(t,:).*log(a3) - (ones(1,K)-Y(t,:)).*log(ones(1,K)-a3))) + lambda/(2*m) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)))
    %sum(-Y(t,:).*log(a3))
    %sum((ones(1,K)-Y(t,:)).*log(ones(1,K)-a3))
    
    %Backward pass
%    delta_3 = a3 - Y(t,:);
%    delta_2 = delta_3*Theta2;
%    delta_2 = delta_2(2:end);
%    delta_2 = delta_2.*sigmoidGradient(z2);
    
%    D2 = D2 + delta_3'*a2;
%    D1 = D1 + delta_2'*a1;
    
%end
%Theta2_grad = 1/m * D2;
%Theta1_grad = 1/m * D1;

%delta3 = Out - Y;
%r2 = delta3*Theta2(:,2:end);
%delta2 = r2.*sigmoidGradient(H1(:,2:end));
%Theta1_grad = 1/m * delta2'*X;
%Theta2_grad = 1/m * delta3'*H1;


%Regularisation terms
t1 = zeros(size(Theta1));
t2 = zeros(size(Theta2));
t1 = [zeros(size(Theta1,1),1), (lambda * Theta1(:,2:end))];
t2 = [zeros(size(Theta2,1),1), (lambda * Theta2(:,2:end))];

%Backward Pass
delta_3 = Out - Y;
delta_2 = (delta_3*Theta2(:,2:end)).*sigmoidGradient(X*Theta1');
%Change gradient for optimisation function.
Theta2_grad = 1/m * (delta_3'*H1 + t2);
Theta1_grad = 1/m * (delta_2'*X + t1);














% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
