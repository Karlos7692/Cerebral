function [TrainData] = maintainNN(NN, TrainData)
    if(strcmp(NN.type, 'srn'))
        TrainData = maintainSRN(NN, TrainData);
    end

end



function [TrainData] = maintainSRN(NN, TrainData)
     lastPlanUnit = NN.shape(1) - sum(NN.state);
     RawData =  TrainData(:, 1:lastPlanUnit);
     TrainData = prepareSRN(NN, RawData);
end
