function [ Xs, Ys ] = bagging(X, Y, nDists)
%BAGGING Summary of this function goes here
%   Detailed explanation goes here
    Xs = cell(nDists,1);
    Ys = cell(nDists,1);
    for i = 1:nDists
        m = size(X,1);
        dist = randi([1,m],m,1);
        Xs{i} = X(dist,:);
        Ys{i} = Y(dist,:);
    end

end

