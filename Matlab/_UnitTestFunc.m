





function results =  _UnitTestFunc()
    printf("********************************************\n");
    printf("            Testing Functionality\n");
    printf("********************************************\n\n");
    
    %Pause Program
    
    printf("Testing Binary Vector Integer Convertions\n");
    r1 = testBinaryVectorIntegerConvertions();
    if r1.fail != 0
         disp ("Failed "), disp(r1.fail), disp("Tests!");
         disp (r1.failed);
    else
        disp("Binary Vector Integer Conversion tests passed!");
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
    while (i <= r.total && failed != 1)
        integer = floor(unifrnd(0,1000000));    
        result = bvecToInt(intToBvec(integer));
        result = bvecToInt(intToPBvec(result));
        if result != integer
            failed = 1;
            r.failed = [integer, r.failed];
            r.fail = r.fail + 1;
        end
        i = i +1;
    end


end


%function r = testReshapeWeights()



