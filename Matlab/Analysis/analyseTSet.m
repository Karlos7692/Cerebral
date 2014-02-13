function [ stats ] = analyseTSet(J_Hist, NN, TrainData, RawOut)
%ANALYSETSET Summary of this function goes here
%   Detailed explanation goes here
figure;
plot(1:length(J_Hist), J_Hist);
title('Learning Costs Over Time');
xlabel('Iterations');
ylabel('Cost');

figure;

%TODO change binary convertions to allow matrix calculations
%TODO change to allow thresholding
%TODO create convertion function
RealPred = zeros(size(RawOut));
BDPred = predict(NN, TrainData);
for i = 1:size(RawOut,1)
    RealPred(i) = bdvecToReal(BDPred(i,:));
end
hold off
plot(1:size(RawOut,1), RawOut, 'r');
hold on
plot(1:size(RealPred, 1), RealPred, 'b');
hold off
title('Stock Pricing Over Time (TrainingSet)');
xlabel('Time (days)');
ylabel('Price ($)');


figure;
plot(1:size(RawOut,1), RawOut - RealPred, 'g');
title('Stock Pricing Difference Over Time');
xlabel('Time (days)');
ylabel('Price ($)');
figure

display('Stock Difference Standard Dev. ($)');
sd = std(RawOut - RealPred);
display(sd);

figure;
scatter(RawOut, RealPred);
lsline;
stats = regstats(RawOut, RealPred, 'linear', {'beta', 'covb', 'yhat', 'r', 'mse', 'rsquare', 'tstat'});
title('Point Correlation');
xlabel('Predicted Value ($)');
ylabel('Actual Value ($)');
display('Correlation Coef. ');
r = sqrt(stats.rsquare);
display(r);
end

