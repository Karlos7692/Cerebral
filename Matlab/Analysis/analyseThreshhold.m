function [tThresh, cvThresh] = analyseThreshhold(NN, TrainData, RawIn, CVIn, RawOut, CVOut)
%ANALYSETHRESHHOLD Summary of this function goes here
%   Detailed explanation goes here
    Total = [RawIn ; CVIn];
    threshs = 0.5:0.005:0.95;
    tthreshSds = zeros(size(threshs));
    cvthreshSds = zeros(size(threshs));
    cvsize = size(CVIn, 1);
    i = 1;
    minTThreshSd = 10000000;
    minCThreshSd = 10000000;
     if(strcmp(NN.type, 'tem'))
        Total = prepareTNN(NN, Total);
    end
    for t = threshs
          trainPred = predict(NN, TrainData, 'reg', t);
          totPred = predict(NN, Total, 'reg', t);
          cvPred = totPred(end-cvsize+1:end, :);
          
          tthreshSds(i) = std(RawOut - trainPred);
          cvthreshSds(i) = std(CVOut - cvPred);
          if (tthreshSds(i) < minTThreshSd)
              minTThreshSd = tthreshSds(i);
              tThresh = t;
          end
          if (cvthreshSds(i) < minCThreshSd)
              minCThreshSd = cvthreshSds(i)
              cvThresh = t;
          end
          i = i + 1;
    end 
    
    plot(threshs, tthreshSds, 'b');
    hold on
    plot(threshs, cvthreshSds, 'g');
    
    
end

