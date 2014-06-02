function [ Distributions ] = bagging(X, Y, nDists)
%BAGGING Summary of this function goes here
%   Detailed explanation goes here
    Distributions = cell(nDists,2);
    for i = 1:nDists
        m = size(X,1);
        dist = randi([1,m],m,1);
        Distributions{i,1} = X(dist,:);
        Distributions{i,2} = Y(dist,:);
    end

end

