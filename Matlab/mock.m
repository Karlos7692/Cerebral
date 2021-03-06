
num_labels = 10;


TData = load('MNIST.mat');
X = TData.X;
y = TData.y;

m = size(X,1);
for i = 1:m
    Y(i,y(i)) = 1;
end

[NN TrainData TargData] = buildNeuralNetwork(X, Y, [100], 0, 'gen', [5,5]);
NN3 = buildNeuralNetwork(X, Y, [10], 0, 'gen', [5,5]);
%Train by Gradient Descent
NN2 = NN;

%%============ Advanced Optimisation =========
fprintf('Training using Advanced Optimisation\n');
lambda = 0.1;

input_layer_size = NN.shape(1);
hidden_layer_size = NN.shape(2);

costFunction = @(p) lmsTest( p, ...
               NN.shape(1), ...
               NN.shape(2), ...
               NN.shape(3), ...
               X, Y, lambda);
           
options = optimset('MaxIter', 200);


[nn_params, cost] = fmincg(costFunction, NN.weights, options);

NN.weights = nn_params;
% Obtain Theta1 and Theta2 back from nn_params
W1 = reshapeWeights(NN,1);
W2 = reshapeWeights(NN,2);


%%============ Gradient Descent ==============
fprintf('Training using gradient descent\n');
alpha1 = 0.0004; 
alpha2 = 0.002;
mew = 0.001;  
[NN2, J_Hist] = gradientDescent(10000, NN2, Y, X, alpha1, mew, lambda);
[NN3, J_Hist2] = gradientDescent(10000, NN3, Y, X, alpha2, mew, lambda);

fprintf('Plotting cost function over time\n');
figure
plot(1:length(J_Hist), J_Hist, 'b');
hold on
plot(1:length(J_Hist2), J_Hist2, 'r');
legend('100-HUS', '10-HUs');
hold off


fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================= Visualize Weights =================
%  You can now "visualize" what the neural network is learning by 
%  displaying the hidden units to see what features they are capturing in 
%  the data.

fprintf('\nVisualizing Neural Networks... \n')
Vis = W1(2:end,:)';
displayData(Vis);
title('Adv Opt.');

figure;
W21 = reshapeWeights(NN2,1);
Vis2 = W21(2:end, :)';
displayData(Vis2);
title('Grad Desc');
fprintf('\nProgram paused. Press enter to continue.\n');
pause;



%% ================= Predict =================
%  After training the neural network, we would like to use it to predict
%  the labels. You will now implement the "predict" function to use the
%  neural network to predict the labels of the training set. This lets
%  you compute the training set accuracy.

pred = predict(NN,TrainData);
[dummy, pred] = max(pred, [], 2);

fprintf('\nTraining Set Accuracy A-Op: %f\n', mean(double(pred == y)) * 100);

pred2 = predict(NN2,TrainData);
fprintf('\nTraining Set Accuracy G-Desc: %f\n', mean(double(pred2 == y)) * 100);















