function [bestThresh] = analyseParticularThreshold(NN, Data, actual)
%ANALYSETRAININGTHRESHOLD Summary of this function goes here
%   Detailed explanation goes here

    threshs = 0.5:0.0005:0.95;
    threshSds = zeros(size(threshs));
    tsd = 10000000;
    i= 1;
    
    bpred = predict(NN, Data, 'reg_no_conv_no_thresh');
    
    for t = threshs
         pred = convertionControl(NN, 'reg', bpred, t);
         threshSds(i) = std(actual - pred);
         if (threshSds(i) < tsd)
              bestThresh = t;
              tsd = threshSds(i);
         end
         i = i+1;
    end
end


function Out = convertionControl(NN, type, Raw, thresh)
     if(strcmp(type,'class'))
         [dummy, Out] = max(Raw, [], 2);
    elseif(strcmp(type, 'reg'))
        Out = convertThreshOutput(NN, Raw, thresh);
    elseif(strcmp(type, 'reg_no_thresh'));
        Out = convertOutput(NN, Raw);
    else
        Out = Raw;
    end


end


function Out = convertThreshOutput(NN, Raw, thresh)

    fon = 1; %fon - first output neuron
    lon = 1; %lon - last output neuron 
    Out = zeros(size(Raw,1), length(NN.output));
    for i= 1:length(NN.output)
        enc = NN.outenc(i);
        lon = lon + NN.output(i) - 1;    %Matlab end = start + n -1;
        Out(:,i) = convertToValueUsingThresh(Raw(:,fon:lon), enc, thresh);      
    end


end

function Out = convertOutput(NN, Raw)

    fon = 1; %fon - first output neuron
    lon = 1; %lon - last output neuron 
    Out = zeros(size(Raw,1), length(NN.output));
    for i= 1:length(NN.output)
        enc = NN.outenc(i);
        lon = lon + NN.output(i) - 1;    %Matlab end = start + n -1;
        Out(:,i) = convertToValue(Raw(:,fon:lon), enc);      
    end


end



function Predictions = convertToValueUsingThresh(Raw, enc, thresh)
    BINARY = 1;
    BINARYDEC = 2;
    Predictions = zeros(size(Raw,1),1);
    for i = 1:size(Raw,1)
        if(enc == BINARY)
            Raw(i,:) = (Raw(i,:) >= thresh);        
            Predictions(i,:) = bvecToInt(Raw(i, :));      
        elseif(enc == BINARYDEC) 
            Raw(i,:) = [(Raw(i,1:end-1) >= thresh), Raw(i,end)];
            Predictions(i,:) = bdvecToReal(Raw(i, :));          
        end        
    end
end


function Predictions = convertToValue(Raw, enc)
    BINARY = 1;
    BINARYDEC = 2;
    Predictions = zeros(size(Raw,1),1);
    for i = 1:size(Raw,1)
        if(enc == BINARY)       
            Predictions(i,:) = bvecToInt(Raw(i, :));      
        elseif(enc == BINARYDEC) 
            Predictions(i,:) = bdvecToReal(Raw(i, :));          
        end        
    end
end