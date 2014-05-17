function [ XSplits ] = transform(ComponentMatrix, Xs)
%TRANSFORMDATA Summary of this function goes here
%   Detailed explanation goes here

    %Concat Xs:
    flen = 0;
    for i = 1:size(Xs, 2)
        flen = flen + size(Xs{i}, 2);
    end
    
    X = zeros(Xs,2);
    fst = 1;
    lst = 0;
    for i = 1:size(Xs,2)
        lst = lst + flen(i);
        X(:,fst:lst) = Xs{i}
        fst = lst + 1;
    end
    
    XSplits = cell(size(X,1), size(ComponentMatrix,2));
    for i = 1:size(ComponentMatrix, 2)
        featVec = ComponentMatrix(:,i);
        XSplit = X(:, featVec);
        XSplits{i} = XSplit;
    end

end

