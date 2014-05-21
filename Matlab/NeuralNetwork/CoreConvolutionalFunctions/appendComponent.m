function [ CNN ] = appendComponent( CNN, Component, ComponentMatrix)
%APPENDCOMPONENT Summary of this function goes here
%   Detailed explanation goes here
    lst = size(CNN.Components,2);
    CNN.Components{lst+1} = Component;
    CNN.ComponentConnections{lst+1} = ComponentMatrix;
    CNN.structure(lst+1) = size(ComponentMatrix, 2);

end

