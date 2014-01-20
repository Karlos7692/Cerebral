












%intvec MUST be a vector.
%Vectors are long row vectors unless otherwise specified.
function bvec = intToBvec(intvec)
    bvec = [];
    for i = 1:length(intvec)
        integer = round(intvec(i));
        bvTemp = [];
        while (integer > 0)
            if mod(integer,2) == 1
                integer = integer - 1;
                bvTemp = [1, bvTemp];     
            else 
                bvTemp = [0, bvTemp];
            end
            integer = integer/2;
        end
        bvec = [bvec, bvTemp];
    end
end






function bvec = intToBvec(intvec, toPad, format)
    %TODO padding for Matrix
    %Constants
     
    %ROW_VEC format appends integer to binary vector conversions as a singular row vector
    ROW_VEC = 1;

    %COL_VEC format appends integer to binary vector conversions as a singular column vector
    %COL_VEC = 2;

    %MATRIX format appends each binary vector conversions as another row in a matrix
    %MATRIX = 3;

    bvec = []
    for i = 1:length(intvec)
        integer = intvec(i);
        while (integer > 0)
            if mod(integer,2) == 1
                integer = integer - 1;
                bvTemp = [1, bvTemp];     
            else 
                bvTemp = [0, bvTemp];
            end
            integer = integer/2;
           
        end
        
        if format == MATRIX
            bvec = [bvTemp;bvec];
        elsif format == COL_VEC
            bvTemp = bvTemp';
            bvec = [bvTemp;bvec];
        else
            bvec = [bvTemp,bvec];
        end
    end
end
