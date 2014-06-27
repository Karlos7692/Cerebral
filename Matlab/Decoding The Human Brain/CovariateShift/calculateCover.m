function [ cover ] = calculateCover(xtrain, XTest, corrThresh)
%CALCULATECOVER Summary of this function goes here
%   Detailed explanation goes here
    cover = 0;
    for i = 1:size(XTest,1)
        %Adjusted Coefficient of determinism
        s = regstats(XTest(i,:), xtrain, 'linear', {'adjrsquare'});
        s.adjrsquare
        if s.adjrsquare > 0
            corr = sqrt(s.adjrsquare);
            if corr > corrThresh
                cover = cover + 1;
            end
        end
    end
    
    cover = cover/size(XTest,1);
    

end

