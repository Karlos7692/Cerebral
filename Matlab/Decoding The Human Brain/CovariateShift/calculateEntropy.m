function [ entropy ] = calculateEntropy( pvec )
%CALCULATEENTROPY Summary of this function goes here
%   E = -p1log(p1) + -p2log(p2) ...

    lpvec = log2(pvec);
    entropy = sum(-1*pvec.*lpvec); 

end

