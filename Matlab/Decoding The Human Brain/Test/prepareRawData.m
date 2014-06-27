function [RawIn, mu, scale] = prepareRawData(X)
%PREPARERAWDATA Summary of this function goes here
%   ComponentMatrix is used by the Convolutional Neural Network to split
%   the Features between the initial Component Neural Networks.
    nCols = size(X,2) * size(X,3);
    nRows = size(X,1);
    RawTemp = zeros(nRows, nCols);
    for i = 1:size(X, 1)
        Scan = squeeze(X(i,:,:));
        feats = reshape(Scan', 1, size(Scan, 1) * size(Scan,2));
        RawTemp(i,:) = feats;
    end
    
    [ mu, scale ] = scaleParam( RawTemp );
    RawTemp = featureScale(RawTemp, mu, scale);
    %Put in cell type consistent types.
    RawIn = cell(1,1);
    RawIn{1} = RawTemp;
   
end

