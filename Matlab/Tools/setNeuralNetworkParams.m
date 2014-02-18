function [NNParams] = setNeuralNetworkParams(hidden , nStateVecs, type, encoding)
%SETNNPARAMS Summary of this function goes here
%   Detailed explanation goes here
    NNParams.type = type;
    NNParams.hidden = hidden;
    NNParams.nStateVecs = nStateVecs;
    NNParams.encoding = encoding;
end

