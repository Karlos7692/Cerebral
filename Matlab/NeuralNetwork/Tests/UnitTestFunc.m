





function results = UnitTestFunc()
    fprintf('********************************************\n');
    fprintf('            Testing Functionality\n');
    fprintf('********************************************\n\n');
    
    %Pause Program
    
    results = [];
   
    fprintf('Testing Binary Vector Integer Convertions\n');
    r = testBinaryVectorIntegerConvertions();
    if r.fail ~= 0
         disp ('Failed '), disp(r1.fail), disp('Tests!');
         disp (r1.failed);
    else
        disp('Binary Vector Integer Conversion tests passed!');
        results = [results, 'passed'];
    end
    pause;
    fprintf('Testing Neural Net Reshaping\n');
    r = testReshapeWeights();
    if(strcmp(r.test, 'failed'))
        disp ('Failed '), disp(r.name), disp('Tests!');
    else
        disp('Testing Neural Net Reshaping passed!');
        results = [results, 'passed'];
    end 
end



%Tests function: binary vector to integer convertion
%Tests 1000 different uniformly drawn integers
function r = testBinaryVectorIntegerConvertions()
    
    %Unit Tests
    r.failed = [];
    r.fail = 0;
    r.total = 1000;
    i = 1;
    failed = 0;
    while (i <= r.total && failed ~= 1)
        integer = floor(unifrnd(0,1000000));    
        result = bvecToInt(intToBvec(integer));
        result = bvecToInt(intToPBvec(result));
        if result ~= integer
            failed = 1;
            r.failed = [integer, r.failed];
            r.fail = r.fail + 1;
        end
        i = i +1;
    end


end


function r = testReshapeWeights()
     r.name = 'Reshape Weights Test';
    
     RawIn = (1:50)';
     RawOut = (2:51)';
     
     [NN TrainData TargData] = buildNeuralNetwork(RawIn, RawOut, 100, 0, 'gen', [1,1]);
     
     weights = NN.weights;
     wsize = size(weights);
     
     W1 = reshapeWeights(NN,1);
     W2 = reshapeWeights(NN,2);
     
     els = numel(W1) + numel(W2);
     
     r.test = 'passed';
     if(els ~= wsize)
         disp('Mismatch number of weights');
         r.test = 'failed';
     end
     
     
     testWeights = [W1(:); W2(:)];
     if(weights ~= testWeights)
         disp('Mismatch number of weights');
         r.test = 'failed';
     end
     
     
end



