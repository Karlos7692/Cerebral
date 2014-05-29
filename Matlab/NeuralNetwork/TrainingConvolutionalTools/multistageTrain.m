function [ CNN, E_Collection ] = multistageTrain(maxIterVec, traintype, costType, CNN, Y, XCell, alpha, mew, lambda)
%MULTISTAGETRAIN Summary of this function goes here
%   Detailed explanation goes here
    trainfunc;
    if strcmp(traintype, 'stoc')
        trainFunc = @stochGradientDescent;
    elseif strcmp(traintype, 'adju')
        trainFunc = @adjStochGradientDescent;
    elseif strcmp(traintype, 'grad')
        trainFunc = @gradientDescent;
    end
    E_Collection = cell(1, length(CNN.Components));
    if CNN.multistage == 1
        for com = 1:length(CNN.Components)
            fprintf('------Training Component %d of %d --------\n');
            
            %Initialize Training Details
            maxIter = maxIterVec(com);
            E_HistMatrix = zeros(maxIter, length(CNN.Components{com}));
            HxCell = cell(length(CNN.Components{com}),1);
            
            %Transform Data into usable format
            [ XSplits ] = transform(CNN.ComponentConnections{com}, XCell);
            for cel = 1:length(CNN.Components{com})
                
                %Train each Neural Network indiviually
                [NN, E_Hist] = trainFunc(maxIter, costType, CNN.Components{com}{cel}, Y,...
                    XSplits{cel}, alpha, mew, lambda);
                CNN.Components{com}{cel} = NN;
                
                %Collect Error History
                E_HistMatrix(:,cel) = E_Hist;
                fprintf('Final Error Component %d, NN%d = %e\n', com, cel, E_Hist(end));
                
                %Get Prediction for next Component:
                Hx = predict(NN, XSplits{cel});
                HxCell{cel} = Hx;
            end
            %Set Input for next component equal to output of trained
            %networks
            XCell = HxCell;
            
            %Save Error Matracies for analysis
            E_Collection{com} = E_HistMatrix;
            fprintf('-------- Component Finished ---------\n');
            
        end
    else
        
    end


end
