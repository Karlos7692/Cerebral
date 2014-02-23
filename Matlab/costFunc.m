function [J, grad] = costFunc(NN, Y, X, lambda)
     [J, grad] = leastMeanSquareTemp(NN, Y, X, lambda);
     %Out(end,:)

end


function [J, grad] = leastMeanSquare(NN, Y, X, lambda)
%TODO Generalise for shape.    
    


    m = size(X,1);
   
    
    W1 = reshapeWeights(NN,1);
    W2 = reshapeWeights(NN,2);
   
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
   
    delta_k = delta_k * W2(2:end,:)'.* (Z2 .* (ones(size(Z2)) - Z2));
    grad_W1 =  (A1' * delta_k + t1);
    
    grad = [grad_W1(:) ; grad_W2(:)];
end



function [J, grad] = leastMeanSquareTemp(NN, Y, X, lambda)
  
    
    m = size(X,1);
    nWeightMatracies = length(NN.shape) - 1;
    nLayers = length(NN.shape);
    %Forward Pass
    %Get Weights Matrix Array
    %Get As, Zs and Hx
    %Get Reglarisation cost
    %Get Regualrisation terms
    %Collect sigmoid gradients
    Weights = cell(1,nWeightMatracies);     %nWs
    Zs = cell(1,nLayers);                   %nZs = nWs
    As = cell(1,nWeightMatracies);          %nAs = nWs
    ts = cell(1,nWeightMatracies);          %nts = nWs
    Zs{1} = X;
    reg = 0;
    sig_grads = cell(1,nWeightMatracies);
    for i = 1:nWeightMatracies
        Weights{i} = reshapeWeights(NN,i);                                       %Get Weights
        As{i} = [ones(m,1), Zs{i}];                                              %Get As
        Zs{i+1} = sigmoid(As{i}*Weights{i});                                     %Get Zs
        reg = reg + (sum(sum(Weights{i}(2:end, :).^2)));                         %Get regularisation cost
        ts{i} = [zeros(1, size(Weights{i},2)); (lambda * Weights{i}(2:end, :))]; %Get regularisation terms
        sig_grads{i} = (Zs{i+1} .* (ones(size(Zs{i+1})) - Zs{i+1}));             %Collect sigmoid gradients (starting layer 2)
    end
    reg = lambda/2 * reg;
    outLayer = length(NN.shape);
    Hx = Zs{outLayer};
    
    %Cost Function
    J =   1/2 * sum(sum((Y - Hx).^2)) + reg;
    
    %Backprop gradient
    grad = zeros(size(NN.weights));                                 %Preallocate grad vector
    lastWeight = numel(NN.weights);                                 %Last weight position for current layer
    delta_k = (-1) * (Y - Hx) .* Hx .* (ones(size(Hx)) - Hx);       %Error signal gradient for outer layer
    
    for i = length(Weights):-1:2                                    %Go backwards from layer n-1:n to 1:2 
        grad_Wi = (As{i}' * delta_k + ts{i});
        firstWeight = lastWeight - numel(grad_Wi) + 1;              %Get first weight position
        grad(firstWeight:lastWeight) = grad_Wi(:);                  %Update Gradient for l-1:l
        lastWeight = lastWeight - numel(grad_Wi);                   %Update Last  Weight Position
        delta_k = delta_k * Weights{i}(2:end,:)' .* sig_grads{i-1}; %Update Error Signal
    end
    %Update first layer - second layer weights
    grad_W1to2 = (As{1}' * delta_k + ts{1});
    grad(1:lastWeight) = grad_W1to2(:);
 
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

