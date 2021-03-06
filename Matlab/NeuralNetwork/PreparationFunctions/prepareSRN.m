function TrainData = prepareSRN(NN, Raw)
    %TODO Put into general form.
    % - Reshape all weight matracies into vector form.
     
    %Pre-allocate weight matrix for SRN
    nStateUs = sum(NN.state);
    m = size(Raw, 1);
    nPlanUs = NN.shape(1) - nStateUs;
    TrainData = zeros(m, NN.shape(1));
    TrainData(:,1:end-nStateUs) = Raw;
        
    TrainData = prepareTrainingData(NN, TrainData);
      
end

function TrainData = prepareTrainingData(NN, TrainData)
   
    firstSVec = 1;
    nStateUs = sum(NN.state);
    fsui = NN.shape(1) - sum(NN.state) + 1;
    
    for i = 1:(size(TrainData,1) - 1)
        Input = TrainData(i:-1:firstSVec, :);
        stateVals = propogateStateValues(NN, Input);
        
        %State vals are transposed to preserve format when
        %converting to vector form. (For readbility.) 
        stateVals = stateVals';
        
        stateVec = stateVals(:);
        nZeroedState = nStateUs - length(stateVec);
        stateVec = stateVec';
        
        if nZeroedState <= 0
             firstSVec = firstSVec + 1;
        else   
             stateVec = [stateVec, zeros(1,nZeroedState)];
        end
        %Change to general form.
        if (fsui <= NN.shape(1))
            TrainData((i+1),fsui:end) = stateVec;
        end
    end
end

%Input consists of plan and state units combined, 
function stateVals = propogateStateValues(NN, Input)
    m = size(Input,1);
    W1 = reshapeWeights(NN,1);
    
    A1 = [ones(m,1), Input];
    stateVals = sigmoid(A1*W1);
end 

