





function Out = predict(NN, Input)

    
    Theta1 = reshapeWeights(NN,1)';
    Theta2 = reshapeWeights(NN,2)';
      
    X = [ones(size(Input,1),1), Input];
    H1 = sigmoid(X*Theta1');
    H1 = [ones(size(H1,1),1), H1];
    Out = sigmoid(H1*Theta2'); 
    
end
