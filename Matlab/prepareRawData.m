function [RawIn, RawOut, CVIn, CVOut] = prepareRawData(Data, outvec, cvsize)
%PREPARERAWINDATA Summary of this function goes here
%   %Data is inputed as the latest day first
%   %All Returned Input Data: RawIn, CVIn,has been feature scaled
%   according to the RawIn data
    CVIn = flipud(Data);
    CVIn = CVIn((end-cvsize+1):end-1, :);
    CVOut = flipud(Data);
    CVOut = CVOut((end-cvsize+2):end, outvec);
    
    RawIn = flipud(Data);
    RawIn = RawIn(1:end-cvsize, :);
    RawOut = flipud(Data(:,outvec));
    RawOut = RawOut(2:end-cvsize+1,:);
   
    [mu scale] = scaleParam(RawIn);
    RawIn = featureScale(RawIn, mu, scale);
    CVIn = featureScale(CVIn, mu, scale);
    
end

