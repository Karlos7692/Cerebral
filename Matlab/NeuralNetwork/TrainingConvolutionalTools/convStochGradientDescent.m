function [ CNN, E_Hist ] = convStochGradientDescent(maxIter, batchSize, costType, CNN, Y, XCell, alpha, mew, lambda)
%CONVSTOCHGRADIENTDESCENT Summary of this function goes here
%   Detailed explanation goes here
    nTraining = size(XCell{1},1);
    nBatches = floor(nTraining/batchSize);
    E_Hist = zeros(maxIter, 1);
    PrevGradCollection = zerosCollection(CNN);

    
    
    %Remainder to redistribute for uneven batch size    
    rem = mod(nTraining, batchSize);    

    for i = 1:maxIter
        
        
        %Extra examples to redistribute
        extra = rem;
        fst = 1;
        lst = 0;
        fprintf(' Iteration %d\t|| \t ', i);
        tic;
        for j = 1:nBatches
             if extra > 0
                %Extra Training examples still left to distribute.
                lst = lst + batchSize + floor(rem/nBatches);
                extra = extra - floor(rem/nBatches);
                
                if j == nBatches
                    lst = lst + extra;
                    extra = 0;
                end
            else
                %No extra training examples to redistribute
                lst = lst + batchSize;
            end
            
            XTemp = XCell{1}(fst:lst,:);
            XTempCell{1} = XTemp;
            [ WeightsCollection, AsCollection, tsCollection, Sig_gradsCollection, HxCollection, reg] = forwardPassComponents(CNN, XTempCell, lambda);
            Hx = HxCollection{end}{end}; %TODO: Change to generalform... Combine and output
            [E, gradconst] = costFunction(Y(fst:lst,:), Hx, reg, costType);
            [GradsCollection] = backwardPassComponents(CNN, WeightsCollection, AsCollection, Sig_gradsCollection, tsCollection, Y(fst:lst,:), HxCollection, gradconst);
            fprintf('\nBatch Size %d -Error- %e \n', (lst-fst+1), E);
            fst = lst + 1;
        
            %Add momentum
            MomCollection = scalarMutltiply(mew, PrevGradCollection);
        
            GradsCollection = scalarMutltiply(alpha, GradsCollection);
        
            %update weights
            [CNN] = subtractGradientAddMomentum(CNN, GradsCollection, MomCollection);
        
            %update momentum
            PrevGradCollection = GradsCollection;
        
            E_Hist(i) = E;
            
        end
        toc;
    end

end


function [Collection] = zerosCollection(CNN)
    Collection = cell(size(CNN.Components));
    for i = 1:length(CNN.Components)     %Collection
         Collection{i} = cell(size(CNN.Components{i}));
         for j = 1:length(CNN.Components{i})  %Cell
             Collection{i}{j} = zeros(size(CNN.Components{i}{j}.weights));
         end
    end

end


function [CNN] = subtractGradientAddMomentum(CNN, GradsCollection, MomCollection)
     for i = 1:length(CNN.Components)     %Collection
         for j = 1:length(CNN.Components{i})  %Cell
             CNN.Components{i}{j}.weights = CNN.Components{i}{j}.weights - GradsCollection{i}{j} - MomCollection{i}{j};
         end
     end
  
end


function [Collection] = scalarMutltiply(scalar, Collection)
    for i = 1:length(Collection)
        for j =  1:length(Collection{i}) %Cell
            Collection{i}{j} = scalar * Collection{i}{j};
        end
    end

end

