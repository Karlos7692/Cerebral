function [ ComponentMatrix ] = generateComponentMatrix(rows, cols)
%GENERATECOMPONENTMATRIX Summary of this function goes here
%   Detailed explanation goes here
     fst=1;  
    %Row vectors refer to the position connections
    %for each NN. 
    %Cols refer to the number neural networks.
    ComponentMatrix = zeros(rows, cols);
    for i = 1:cols
        lst = i*rows;
        vec = (fst:lst)';
        ComponentMatrix(:,i) = vec;
        fst=lst+1;
    end

end

