function [ HxCell ] = convPredict( CNN, Input )
%CONVPREDICT Summary of this function goes here
%   Detailed explanation goes here
    for i=1:length(CNN.structure)
        XSplits = transform(CNN.ComponentConnections{i}, Input);
        HxCell = cell(size(CNN.Components{i}));
        for j = 1:length(CNN.Components{i})
            Hx = predict(CNN.Components{i}{j}, XSplits{j});
            HxCell{j} = Hx;
        end
        Input = HxCell;
    end

end

