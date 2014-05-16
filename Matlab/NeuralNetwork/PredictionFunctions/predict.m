





function Out = predict(NN, Input)
    %TODO Put in general form and remove unecessary transpositions.
    
    %Theta1 = reshapeWeights(NN,1)';
    %Theta2 = reshapeWeights(NN,2)';
      
    %X = [ones(size(Input,1),1), Input];
    %H1 = sigmoid(X*Theta1');
    %H1 = [ones(size(H1,1),1), H1];
    %Out = sigmoid(H1*Theta2');
    
    
    
    nWeightMatracies = length(NN.shape) - 1;
    Z = Input;
    for i = 1:nWeightMatracies
        A = [ones(size(Z,1),1), Z];
        W = reshapeWeights(NN,i);
        Z = sigmoid(A*W);
    end
    Out = Z;
    
    
    %TODO Put in seperate function.
    %Z is final output
    
   % if(strcmp(type,'class'))
   %      [dummy, Out] = max(Z, [], 2);
   % elseif(strcmp(type, 'reg'))
   %     Out = convertThreshOutput(NN, Z, thresh);
   % elseif(strcmp(type, 'reg_no_thresh'));
   %     Out = convertOutput(NN, Z);
   % else
   %     Out = Z;
   % end
    
    
    
    
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