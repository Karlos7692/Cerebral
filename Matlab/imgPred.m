num_labels = 10;


TData = load('MNIST.mat');
X = TData.X;
y = TData.y;

m = size(X,1);
for i = 1:m
    Y(i,y(i)) = 1;
end
layers = [50,50,25];
[NN TrainData TargData] = buildNeuralNetwork(X, Y, layers, 0, 'gen', [5,5]);

alpha = 0.0003;
mew= 0;
lambda = 0;

[NN, J_Hist] = gradientDescent(10000, NN, Y, X, alpha, mew, lambda);

plot(1:length(J_Hist), J_Hist);
title('Error over Time');
xlabel('Iterations');
ylabel('LMS Cost');


pred = predict(NN, TrainData, 'class', 0);
fprintf('\nTraining Accuracy of %d layered Neural Network: %f\n', length(layers) ,mean(double(pred == y)) * 100);
