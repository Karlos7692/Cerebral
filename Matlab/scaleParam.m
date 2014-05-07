function [ mu scale ] = scaleParam( Data )
%SCALEPARAM Summary of this function goes here
%   Detailed explanation goes here
   mu = mean(Data);
   scale = max(Data) - min(Data)

end

