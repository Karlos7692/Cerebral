function [CVPred] = cvanalyser(NN, RawIn, CVIn, CVOut)
%CVANALYSER Summary of this function goes here
%   Detailed explanation goes here
    hold off;
    f1 = figure;
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
    

end

