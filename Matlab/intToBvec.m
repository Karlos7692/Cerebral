












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



