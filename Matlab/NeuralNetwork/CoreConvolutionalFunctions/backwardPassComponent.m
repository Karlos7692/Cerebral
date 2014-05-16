function [ Grads, Delta_ks ] = backPassComponent(CNN, delta_k, WeightsCell, AsCell, Sig_gradsCell, tsCell, gc, index)
%BACKPASSCOMPONENTS Summary of this function goes here
%   Detailed explanation goes here
    
    Grads = cell(length(CNN.Components{index}),1);
    Delta_ks = cell(length(CNN.Components{index}),1);
    for i = 1:length(CNN.Components{index})
        [grad, delta_k ] = bakwardPassError(delta_k, CNN.Components{index}{i}, WeightsCell{i},AsCell{i}, Sig_gradsCell{i}, tsCell{i}, gc)
        Grads{i} = grad;
        Delta_ks{i} = delta_k;
    end
    %Grads will be subtracted in convGradientDescent.
    %Delta_Ks vectors will also be combined in convGradientDescent.
    
end

