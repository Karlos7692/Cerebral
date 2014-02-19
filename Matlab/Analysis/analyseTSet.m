function [stats, bestThresh] = analyseTSet(J_Hist, NN, TrainData, RawOut, threshType)
%ANALYSETSET Summary of this function goes here
%   Detailed explanation goes here
figure;
plot(1:length(J_Hist), J_Hist);
title('Learning Costs Over Time');
xlabel('Iterations');
ylabel('Cost');
fprintf('Final Cost: %f\n', J_Hist(end));
figure;

%TODO change binary convertions to allow matrix calculations
%TODO change to allow thresholding
%TODO create convertion function
bestThresh = 0;
if (~strcmp(threshType, 'reg_no_thresh'))
    bestThresh = analyseParticularThreshold(NN, TrainData, RawOut);
end
RealPred = predict(NN,TrainData, threshType, bestThresh);
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
title('Stock Pricing Difference Over Time (Training Set)');
xlabel('Time (days)');
ylabel('Price ($)');


sd = std(RawOut - RealPred);
fprintf('Training Set: Stock Difference Standard Dev. $%f\n',sd);

figure;
scatter(RawOut, RealPred);
lsline;
stats = regstats(RawOut, RealPred, 'linear', {'beta', 'covb', 'yhat', 'r', 'mse', 'rsquare', 'tstat'});
title('Point Correlation (Training Set)');
xlabel('Predicted Value ($)');
ylabel('Actual Value ($)');
r = sqrt(stats.rsquare);
fprintf('Correlation Coef: %f\n\n', r);
end

