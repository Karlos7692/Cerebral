function [ DistWeights ] = covariateShift(Dists, Outs, nFeats, nSamples)
%COVARIATESHIFT Summary of this function goes here
%   Detailed explanation goes here
    
    %Change Distribution to Rate of Change per feature for scale invarience.
   % fprintf('Calculating Rate of Change...\n');
   % for i = 1:length(Dists)
   %     X = Dists{i};
   %     fst = 1;
   %     XChange = zeros(size(X));
   %     tic;
   %     for j = 1:nFeats
   %         lst = j*nSamples;
   %         FeatX = X(:, fst:lst);
   %         FeatChange = calculateRateChange( FeatX );
   %         XChange(:, fst:lst) = FeatChange;
   %         fst = lst +1;
   %     end
   %     Dists{i} = XChange;
   %     toc;
   % end
    
    
    fprintf('Learning Distribution Weights... \n');
    %Using The rate change,get distribution weights
    DistWeights = cell(size(Dists));
    for subject = 1:length(Dists)
        [XTest,YTest] = concatDists( Dists, Outs, subject);
        tic;
        [ distWeights, avgEntropy ] = learnDistributionWeights(Outs{subject}, Dists{subject}, YTest, XTest);
        toc;
        DistWeights{subject} = distWeights;
        fprintf('Average Entropy %f\n', avgEntropy);
    end

end

