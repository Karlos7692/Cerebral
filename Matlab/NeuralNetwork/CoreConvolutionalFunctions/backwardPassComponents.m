function  [GradsCollection] = backwardPassComponents(CNN, WeightsCollection, AsCollection, Sig_gradsCollection, tsCollection, Y, HxCollection, gc)
%BACKWARDPASSCOMPONENTS Summary of this function goes here
%   Detailed explanation goes here
    
    %Original BackComponent: Change to General Form.
    Hx = HxCollection{end}{1};
    Delta_k = (-1) * (Y - Hx);     
    Err_Sigs{1} = Delta_k;
    %Other Components
    
    
    for i = length(CNN.structure):-1:1

        [Grads, Err_Sigs ] = backwardPassComponent(CNN, Err_Sigs, WeightsCollection{i}, AsCollection{i}, Sig_gradsCollection{i}, tsCollection{i}, HxCollection{i}, gc, i);
        GradsCollection{i} = Grads;
    end

end

