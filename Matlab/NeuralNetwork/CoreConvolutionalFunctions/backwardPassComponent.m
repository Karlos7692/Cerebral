function [ Grads, Delta_ks ] = backPassComponent(CNN, Delta_ks, WeightsCell, AsCell, Sig_gradsCell, tsCell, gc, index)
%BACKPASSCOMPONENTS Summary of this function goes here
%   Detailed explanation goes here
    
    Grads = cell(length(CNN.Components{index}),1);
    transform(CNN.Inverted
    Delta_ks = cell(length(CNN.Components{index}),1);
    for i = 1:length(CNN.Components{index})
        [grad, delta_k ] = bakwardPassError(delta_k, CNN.Components{index}{i}, WeightsCell{index}{i},...
        AsCell{index}{i}, Sig_gradsCell{index}{i}, tsCell{index}{i}, gc);
        Grads{i} = grad;
        Delta_ks{i} = delta_k;
    end
    %Grads will be subtracted in convGradientDescent.
    %Delta_Ks vectors will also be combined in convGradientDescent.
    
end

