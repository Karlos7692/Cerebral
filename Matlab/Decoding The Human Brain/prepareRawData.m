function [ RawIn, ComponentMatrix ] = prepareRawData(X)
%PREPARERAWDATA Summary of this function goes here
%   ComponentMatrix is used by the Convolutional Neural Network to split
%   the Features between the initial Component Neural Networks.
    nCols = size(X,2) * size(X,3);
    nRows = size(X,1);
    RawIn = zeros(nRows, nCols);
    for i = 1:size(X, 1)
        Scan = squeeze(X(i,:,:));
        feats = reshape(Scan', 1, size(Scan, 1) * size(Scan,2));
        RawIn(i,:) = feats;
    end
end

