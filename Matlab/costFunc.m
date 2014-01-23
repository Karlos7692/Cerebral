function [J, grad] = costFunc(NN, Y, X, lamda)
     [J, grad] = leastMeanSquare(NN, Y, X, lamda);
     %Out(end,:)

end


function [J, grad] = leastMeanSquare(NN, Y, X, lamda)
%TODO Generalise for shape.    
    


    m = size(X,1);
    K = size(Y,2);
    
    W1 = reshapeWeights(NN,1);
    W2 = reshapeWeights(NN,2);
    
    A1 = [ones(m,1), X];
    Z2 = sigmoid(A1*W1);
    A2 = [ones(m,1), Z2];
    Hx = sigmoid(A2*W2);
    
    J =   1/2 * sum(sum((Y - Hx).^2));
    
    %Backprop gradient
    %delta_k = dE/dOut 
    delta_k = (Y - Hx) .* Hx .* (ones(size(Hx)) - Hx);
    grad_W2 =  1/m * A2' * delta_k;
    
    %delta_k = dE/dHidden
    delta_k = delta_k * W2(2:end, :)' .* (Z2 .* (ones(size(Z2)) - Z2));
    grad_W1 =  1/m * A1' * delta_k;
    
    
    grad = [grad_W1(:) ; grad_W2(:)];
end



function [J, grad, Out] = crossEntropyCost(NN, Y, X, lambda)

%TODO Get rid of transpose
%TODO Generalise for shape.

%    A = X;
%    for i = 1:(length(NN.shape)-1)
%        W = reshapeWeights(NN,i);
%        Z = [ones(size(A,1)), A];
%        A = sigmoid(Z*W);
%    end
%    Hx = A;
%    ce =  1/m * sum(-Y.*log(Hx) - (ones(size(Y))-Y).*log(ones(size(Y))-Hx));
    
    %Find Regularisation cost
    %Add regularisation cost to cross entropy cost.
    m = size(X,1);
    K = size(Y,2);
    
    Theta1 = reshapeWeights(NN,1)';
    Theta2 = reshapeWeights(NN,2)';
    
  
    X = [ones(m,1), X];
    H1 = sigmoid(X*Theta1');
    H1 = [ones(m,1), H1];
    Out = sigmoid(H1*Theta2'); 
    
    
    J = 1/m* sum(sum(-Y.*log(Out) - (ones(m,K)-Y).*log(ones(m,K)-Out))) + lambda/(2*m) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));
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
    
    Theta1_grad = Theta1_grad';
    Theta2_grad = Theta2_grad';
    grad = [Theta1_grad(:) ; Theta2_grad(:)];
end

