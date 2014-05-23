function [ CNN, E_Hist ] = convGradientDescent(maxIter, costType, CNN, Y, XCell, alpha, mew, lambda)
%CONVGRADIENTDESCENT Summary of this function goes here
%   Detailed explanation goes here
    E_Hist = zeros(maxIter, 1);
    PrevGradCollection = zerosCollection(CNN);

    for i = 1:maxIter
        %Neural Network maintenance: Set correct TData

        
        [ WeightsCollection, AsCollection, tsCollection, Sig_gradsCollection, HxCollection, reg] = forwardPassComponents(CNN, XCell, lambda);
        Hx = HxCollection{end}{end}; %TODO: Change to generalform... Combine and output
        [E, gradconst] = costFunction(Y, Hx, reg, costType);
        [GradsCollection] = backwardPassComponents(CNN, WeightsCollection, AsCollection, Sig_gradsCollection, tsCollection, Y, HxCollection, gradconst);
        
        
        %Add momentum
        MomCollection = scalarMutltiply(mew, PrevGradCollection);
        
        GradsCollection = scalarMutltiply(alpha, GradsCollection);
        
        %update weights
        [CNN] = subtractGradientAddMomentum(CNN, GradsCollection, MomCollection);
        
        %update momentum
        PrevGradCollection = GradsCollection;
        
        E_Hist(i) = E;
        
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
             CNN.Components{i}{j}.weights = CNN.Components{i}{j}.weights - GradsCollection{i}{j} + MomCollection{i}{j};
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

