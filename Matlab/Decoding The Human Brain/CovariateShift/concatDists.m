function [ XTest,YTest] = concatDists( Dists, Outs, index )
%CONCATDISTS Summary of this function goes here
%   Detailed explanation goes here

    %Todo preallocatie returned values
    XTest = [];
    YTest = [];
    for i = 1:length(Dists)
        if i ~= index
            XTest = [XTest;Dists{i}];
            YTest = [YTest;Outs{i}];
        end
    end
end

