function [J10, J100] = MockImageRec(nntype, nStatevecs, mom)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    maxIter = 20;
    
    J10LR = 0.0003:0.0003:(maxIter*0.0003);
    J100LR = 0.00003:0.00003:(maxIter*0.00003);
    
    TData = load('MNIST.mat');
    X = TData.X;
    y = TData.y;
    m = size(X,1);
    for i = 1:m
        Y(i,y(i)) = 1;
    end
    
    %shuffle data accordingly.
    perm = randperm(m);
    RawIn = X(perm,:);
    RawOut = Y(perm,:);
   
    
    for i = 1:maxIter
        figure;
        [NN, TrainData, TargData] = buildNeuralNetwork(RawIn, RawOut, 10, nStatevecs, nntype, [5,5]);
        [NN, J_Hist] = gradientDescent(10000, NN, TargData, TrainData, J10LR(i), mom, 0);
        hold on
        J10(i) = J_Hist(end);
        plot(1:length(J_Hist), J_Hist, 'b');
        [NN, TrainData, TargData] = buildNeuralNetwork(RawIn, RawOut, 100, nStatevecs, nntype, [5,5]);
        [NN, J_Hist] = gradientDescent(10000, NN, TargData, TrainData, J100LR(i), mom, 0);
        J100(i) = J_Hist(end);
        plot(1:length(J_Hist), J_Hist, 'r');
        legend('10 HUs', '100 HUs');
        xlabel('Epochs');
        ylabel('LMS Error');
        title(strcat('10 HUs vs 100 HUs, LR-10 = ', num2str(J10LR(i)), ' LR-100 = ', num2str(J100LR(i))));     
    end
    
    figure
    xlabel('Learning Rate');
    ylabel('LMS Error');
    title('Cost per Learning Rate for 10/100 HUs');
    plot(J10LR, J10, 'b', 'Marker', '+');
    hold on
    plot(J100LR, J100, 'g', 'Marker', '+');
    legend('10 HUs', '100 HUs');
    hold off
end

