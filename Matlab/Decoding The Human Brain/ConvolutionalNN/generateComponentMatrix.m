function [ ComponentMatrix ] = generateComponentMatrix(nFeats, nNNs)
%GENERATECOMPONENTMATRIX Summary of this function goes here
%   Detailed explanation goes here
    fst=1;  
    %Row vectors refer to the position connections
    %for each NN. 
    %Cols refer to the number neural networks.
    ComponentMatrix = zeros(nFeats, nNNs);
    for i = 1:nNNs
        lst = i*nFeats;
        vec = (fst:lst)';
        ComponentMatrix(:,i) = vec;
        fst=lst+1;
    end

end

