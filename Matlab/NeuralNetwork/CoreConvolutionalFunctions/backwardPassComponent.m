function [ Grads, Delta_ks ] = backwardPassComponent(CNN, Err_Sigs, WeightsCell, AsCell, Sig_gradsCell, tsCell, gc, index)
%BACKPASSCOMPONENTS Summary of this function goes here
%   Detailed explanation goes here
    
    Grads = cell(length(CNN.Components{index}),1);
    
    %Combine Gradients:
    %Delta_k resembles the combined error signals passed back to its
    %inputs. Since all backpropogation from previous component have the
    %assumption of total interconnection, and the data itself is
    %transformed to partition connections between components.
    Delta_k = combine(CNN.ComponentConnections{index}, Err_Sigs);
    
    %Assume the Component Neural Networks have the same number of outputs
    nOuts = CNN.Compontents{index}{1}.shape(end);
    fst = 1;
    
    Delta_ks = cell(length(CNN.Components{index}),1);
    for i = 1:length(CNN.Components{index})
        lst = i*nOuts;
        
        %For each Neural Network pass back the assigned Combined Errors.
        %Transformation T: C_i.O -> C_i+1.I is in direction of Forward Pass.
        %Therefore delta_k for C_i.O_k = Sum j=k DC.I_j
        [grad, delta_k ] = bakwardPassError(Delta_k(:,fst:lst), CNN.Components{index}{i}, WeightsCell{index}{i},...
        AsCell{index}{i}, Sig_gradsCell{index}{i}, tsCell{index}{i}, gc);
        
        fst = lst +1;
    
        Grads{i} = grad;
        Delta_ks{i} = delta_k;
        
    end
    
    %Grads will be subtracted in convGradientDescent.
    %Delta_Ks vectors will also be combined in convGradientDescent.
    
end

