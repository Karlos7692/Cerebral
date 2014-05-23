function [ ApproxComponents ] = gradApproxCNN( CNN, XCell, costType)
%GRADAPPROXTEST Summary of this function goes here
%   Detailed explanation goes here
    
    %For all neural networks in all cells in all components. Peturb high
    %and low for each weight. Calculate line betwwen high and low points to
    %approximate partial differentiation.
    e = 1e-4;
    ApproxComponents = cell(size(CNN.Components));
    for com = 1:length(CNN.Components)  %com - component interator
        
        ApproxCell = cell(size(CNN.Components{com}));
        for cel = 1:length(CNN.Components{com}) %cel - cell iterartor
            
            approx = zeros(size(CNN.Components{com}{cel}.weights));
            peturb = zeros(size(CNN.Components{com}{cel}.weights));
            for p = 1:length(CNN.Components{com}{cel}.weights) %p - NN.weights position iterator
                    
                %Create High and Low CNN
                CNNHigh = CNN; %Copies CNN;
                CNNLow = CNN;
                    
                %Assign peturbation vector to a particular neural
                %network
                peturb(p) = e;
                NNHigh = CNNHigh.Components{com}{cel};
                NNLow = CNNLow.Components{com}{cel};
                NNHigh.weights = NNHigh.weights + peturb;
                NNLow.weights = NNHigh.weights - peturb;
                    
                %Assign H-L NNs to particular CNN[H|L]-Component-Cell-NN
                CNNHigh{com}{cel} = NNHigh;
                CNNLow{com}{cel} = NNLow;  %hi juttie waz here
                    
                    
                %Get Prediction 
                [ HxCellHigh ] = convPredict( CNNHigh, XCell );
                [ HxCellLow ] = convPredict( CNNLow, XCell );
                
                %Unpackage Prediction
                HxHigh = HxCellHigh{end};
                HxLow = HxCellLow{end};
                    
                %Get Regaularisation for particular
                    
                    
                %Calculate Cost for particular Cost Function Type.
                [ E2 ] = costFunction(Y, HxHigh, regHigh, costType);
                [ E1 ] = costFunction(Y, HxLow, regLow, costType);
                    
                %Approx Grad For NN
                approx(p) = (E2 - E1) / (2*e);
                
                %Reset Peturb Vector
                peturb(p) = 0;
            end
            ApproxCell{cell} = approx;
        end
        ApproxComponents{com} = ApproxCell;
    end   
end

