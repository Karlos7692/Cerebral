function [ CombinationMatrix ] = combine(ComponentMatrix, Ks)
%COMBINE Summary of this function goes here
%   Detailed explanation goes here
    
    
    %Ks are a cell;
    %Concat Xs:
    flen = 0;
    for i = 1:size(Ks, 2)
        flen = flen + size(Ks{i}, 2);
    end
    
    K = zeros(size(Ks{1},1),flen);
    fst = 1;
    lst = 0;

    for i = 1:size(Ks,1)
        lst = lst + size(Ks{i}, 2);
        K(:,fst:lst) = Ks{i};
        fst = lst + 1;
    end
    
    %Add all vectors with the same position in the ComponentMatrix
    %Pre-Allocate Size of Combination Matrix
    lstvec = max(max(ComponentMatrix));
    nNNs = size(ComponentMatrix,2);
    CombinationMatrix = zeros(size(K,1), lstvec);

    for i = 1:nNNs
        for j = 1:size(ComponentMatrix,1)
            neuron = ComponentMatrix(j,i);
            errPos = (i-1)*size(ComponentMatrix,1) + j;
            errorvec = K(:,errPos);
            CombinationMatrix(:,neuron) = CombinationMatrix(:,neuron) + errorvec;
        end
    end

end

