function [stats] = analyseCVSim(CVPred, CVOut)
%CVANALYSER Summary of this function goes here
%   Detailed explanation goes here
    hold off;

    %Plot Predicted values overlayed Actual Values
    plot(1:length(CVOut), CVOut, 'r');
    hold on
    plot(1:length(CVOut), CVPred, 'b');
    xlabel('Time');
    ylabel('Stock Price');
    title('Cross Validation Comparision');
    hold off
     
 
    figure;  
    scatter(CVPred, CVOut);
    title('Actual vs Prediction Scatter, (CV Set)');
    xlabel('Prediction ($)');
    ylabel('Actual ($)');
    lsline
    stats = regstats(CVOut, CVPred, 'linear', {'beta', 'covb', 'yhat', 'r', 'mse', 'rsquare', 'tstat'});  
    r = sqrt(stats.rsquare);
    fprintf('CV Correlation Coef: %f\n', r);
    
    figure;
    plot(1:length(CVOut), CVOut - CVPred, 'g');
    title('Difference between Actual and Predicted');
    xlabel('Time');
    ylabel('Stock Price');
    sd = std(CVOut - CVPred);
    fprintf('CV Std. %f\n', sd);
   
    
end