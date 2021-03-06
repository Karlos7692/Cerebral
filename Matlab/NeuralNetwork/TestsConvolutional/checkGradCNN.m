function [passed, failed, totalTests, GradsCollection, ApproxCollection] = checkGradCNN(CNN, Y, XCell, lambda, costType)
%CHECKGRADCNN Summary of this function goes here
%   Detailed explanation goes here

    [ WeightsCollection, AsCollection, tsCollection, Sig_gradsCollection, HxCollection, reg] = forwardPassComponents(CNN, XCell, lambda);
    Hx = HxCollection{end}{end}; %TODO: Change to generalform... Combine and output
    [E, gradconst] = costFunction(Y, Hx, reg, costType);
    [GradsCollection] = backwardPassComponents(CNN, WeightsCollection, AsCollection, Sig_gradsCollection, tsCollection, Y, HxCollection, gradconst);
    
    
    [ ApproxCollection ] = gradApproxCNN(CNN, Y, XCell, lambda, costType);
    
    passed = zeros(size(CNN.structure));
    failed = zeros(size(CNN.structure));
    totalTests = CNN.structure;
    %Append To One Larger Test:
    fprintf('--------- Gradient Test ---------\n');
    len = 0;
    for com = 1:length(CNN.Components)     
        for cel = 1:length(CNN.Components{com})
            len = len + length(CNN.Components{com}{cel}.weights);
        end
    end
    gradvec = zeros(len,1);
    approxvec = zeros(len,1);
    fst = 1;
    lst = 0;
    for com = 1:length(CNN.Components)     
        for cel = 1:length(CNN.Components{com})
            lst = lst + length(GradsCollection{com}{cel});
            gradvec(fst:lst) = GradsCollection{com}{cel};
            approxvec(fst:lst) = ApproxCollection{com}{cel};
            fst = lst + 1;
            
        end
    end
    
    
    diff = norm(approxvec-gradvec)/norm(approxvec+gradvec);
    if diff < 1e-8
         fprintf('Test Passed! Component: %d, NN: %d, Rel Diff: %e\n', com, cel, diff);
    else
         fprintf('Test Failed! Component: %d, NN: %d, Rel Diff: %e\n', com, cel, diff);
    end
     
end

