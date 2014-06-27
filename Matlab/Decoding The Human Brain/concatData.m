function [ X,Y ] = concatData( Xs, Ys, includevec)
%CONCATDATA Summary of this function goes here
%   Detailed explanation goes here

    X = [];
    Y = [];
    for include = includevec
        X = [X;Xs{include}];
        Y = [Y;Ys{include}];
    end

end

