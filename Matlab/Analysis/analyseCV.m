function [CVPred, stats] = analyseCV(NN, RawIn, CVIn, CVOut)
%CVANALYSER Summary of this function goes here
%   Detailed explanation goes here
    hold off;
    cvsize = size(CVIn, 1);
    
    Total = [RawIn ; CVIn];
    
    if(strcmp(NN.type, 'tem'))
        Total = prepareTNN(NN, Total);
    end
    
    CVPred = zeros(size(CVOut));
    TotalPred = predict(NN, Total);
    
    CVBPreds = TotalPred(end-cvsize+1:end, :);
    
    
    %TODO change binary convertions to allow matrix calculations
    %TODO change to allow thresholding
    %TODO create convertion function
    for i = 1:size(CVOut,1)
        CVPred(i) = bdvecToReal(CVBPreds(i,:));
    end
    
    %Plot Predicted values overlayed Actual Values
    plot(1:length(CVOut), CVOut, 'r');
    hold on
    plot(1:length(CVOut), CVPred, 'b');
    plot(1:length(CVOut), CVOut - CVPred, 'g');
    xlabel('Time');
    ylabel('Stock Price');
    title('Cross Validation Comparision');
    hold off
    
    figure;
    title('Actual vs Prediction, (CV Set)');
    xlabel('Prediction ($)');
    ylabel('Actual ($)');
    scatter(CVPred, CVOut);
    lsline
    stats = regstats(CVOut, CVPred, 'linear', {'beta', 'covb', 'yhat', 'r', 'mse', 'rsquare', 'tstat'});
    display('CV Correlation Coef. ');
    r = sqrt(stats.rsquare);
    display(r);
    
    
    figure;
    title('CV Corrections');
    xlabel('CV Prediction with Corrections');
    ylabel('Actual Value');
    plot(1:length(CVOut), CVOut, 'r');
    hold on
    plot(1:length(CVOut), stats.yhat, 'b');
     plot(1:length(CVOut), CVOut - stats.yhat, 'g');
    display('CV LR. Correction Std. ');
    sd = std(CVOut - stats.yhat);
    display(sd);
    hold off
end

