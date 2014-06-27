function [ RateChangeSignal ] = calculateRateChange( Signals )
%CALCULATERATECHANGE Summary of this function goes here
%   Detailed explanation goes here

    RateChangeSignal = zeros(size(Signals));
    for i = 1:size(Signals,1);
        epsilon = 1e-18;
        delta_s = zeros(1, size(Signals,2));
        for j = 2:length(delta_s)
            if Signals(i,j-1) ~= 0
                delta_s(1,j) = (Signals(i,j)-Signals(i,j-1))/Signals(i,j-1);
            else
                %Signal(i,j-1) undefined. Introduce margin of
                %error.
                delta_s(1,j) = (Signals(i,j)-sign(Signals(i,j))*epsilon)/epsilon;
            end
        end
        RateChangeSignal(i,:) = delta_s;
    end
end

