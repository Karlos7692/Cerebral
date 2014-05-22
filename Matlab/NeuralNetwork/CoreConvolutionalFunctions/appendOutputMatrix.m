function [ CNN ] = appendOutputMatrix(CNN, nOuts)
%APPENDOUTPUTMATRIX Summary of this function goes here
%   Detailed explanation goes here
    ComponentMatrix = generateComponentMatrix(nOuts,1);
    lst = size(CNN.Components,2);
    CNN.ComponentConnections{lst+1} = ComponentMatrix;
end

