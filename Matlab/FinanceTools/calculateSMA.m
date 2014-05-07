function [ SMA ] = calculateSMA( Returns, freq)
%CALCULATESMA Summary of this function goes here
%   Detailed explanation goes here
    
    SMA = zeros(size(Returns))
    for i = freq:size(Returns, 2):
        s = sum(Returns(i -freq +1:freq, 2) %Matlab Indexing Starts at 1
        SMA(i,:) = s./freq
    end
    
    
end

