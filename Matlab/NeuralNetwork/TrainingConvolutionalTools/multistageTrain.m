function [ CNN, E_Collection ] = multistageTrain(maxIterVec, traintype, costType, batchSize, CNN, Y, XCell, alphavec, mewvec, lambdavec)
%MULTISTAGETRAIN Summary of this function goes here
%   Detailed explanation goes here
    nBatches = floor(size(XCell{1},1)/batchSize);
    E_Collection = cell(1, length(CNN.Components));
    if CNN.multistageTraining == 1
        for com = 1:length(CNN.Components)
            fprintf('------Training Component %d of %d --------\n', com, length(CNN.Components));
            
            %Initialize Training Details
            maxIter = maxIterVec(com);
            E_HistMatrix = zeros(maxIter * nBatches, length(CNN.Components{com}));
            HxCell = cell(length(CNN.Components{com}),1);
            alpha = alphavec(com);
            mew = mewvec(com);
            lambda = lambdavec(com);
            
            %Transform Data into usable format
            [ XSplits ] = transform(CNN.ComponentConnections{com}, XCell);
            for cel = 1:length(CNN.Components{com})
                
                %Train each Neural Network indiviually
                tic;
                
                 
                if strcmp(traintype, 'stoc')
                   [NN, E_Hist] = stocGradientDescent(maxIter, batchSize, costType, CNN.Components{com}{cel}, Y, XSplits{cel}, alpha, mew, lambda);
                elseif strcmp(traintype, 'adju')
                 
                else
                   [NN, E_Hist] = gradientDescent(maxIter, costType,CNN.Components{com}{cel}, Y, XSplits{cel}, alpha, mew, lambda);
                end
                
                CNN.Components{com}{cel} = NN;
                
                %Collect Error History
                E_HistMatrix(:,cel) = E_Hist;
                fprintf('Final Error Component %d, NN%d = %e\t || \t', com, cel, E_Hist(end));
                toc;
                
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

