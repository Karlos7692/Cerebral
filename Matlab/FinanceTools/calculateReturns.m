function [ Returns ] = calculateReturns(Prices)
%CALCULATERETURNS Summary of this function goes here
%   Prices are a matrix whereby each column vector is a particular price
    Returns = zeros(size(Prices));
    for i = 2:size(Prices,2)
        Returns(i,:) = (Prices(i,:) - Prices(i,:))./Prices(i-1,:);
    end

end

