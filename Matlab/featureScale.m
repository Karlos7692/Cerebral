function Data = featureScale( Data )
%FEATURESCALE Summary of this function goes here
%   Detailed explanation goes here
   mu = mean(Data);
   scale = max(Data) - min(Data);
   
   for i = 1:size(Data , 2)
       Data(:,i) = (Data(:, i) - mu(i))/scale(i);
   end
end

