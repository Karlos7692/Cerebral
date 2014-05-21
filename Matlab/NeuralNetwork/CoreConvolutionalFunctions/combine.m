function [ CombinationMatrix ] = combine(ComponentMatrix, Ks)
%COMBINE Summary of this function goes here
%   Detailed explanation goes here
    
    %Ks are a cell;
    %Concat Xs:
    flen = 0;
    for i = 1:size(Ks, 2)
        flen = flen + size(Ks{i}, 2);
    end
    
    K = zeros(Ks,2);
    fst = 1;
    lst = 0;
    for i = 1:size(Ks,2)
        lst = lst + flen(i);
        K(:,fst:lst) = Ks{i};
        fst = lst + 1;
    end
  
    %Add all vectors with the same position in the ComponentMatrix
    %Pre-Allocate Size of Combination Matrix
    lstvec = max(max(ComponentMatrix));
    nnSize = size(ComponentMatrix,2);
    CombinationMatrix = zeros(size(K,1), lstvec);
    
    for i = 1:size(ComponentMatrix,1)
        for j = 1:size(ComponentMatrix,2)
            vecPos = ComponentMatrix(i,j);
            kvecPos = (j - 1)*nnSize + i;    %vecPos equivalent vector is kvePos
            kvec = K(:,kvecPos);         
            CombinationMatrix(:,vecPos) = CombinationMatrix(:,vecPos) + kvec;
        end
    end

end

