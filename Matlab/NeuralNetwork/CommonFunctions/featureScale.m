function Data = featureScale( Data, mu, scale )
%FEATURESCALE Summary of this function goes here
%   Detailed explanation goes here  
   for i = 1:size(Data , 2)
       Data(:,i) = (Data(:, i) - mu(i))/scale(i);
   end
end

