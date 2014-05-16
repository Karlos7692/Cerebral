











function Converted = convertData(Raw, type)
    %Types:
    %BINARY type encodes integer values as a binary output
    BINARY = 1;

    %BINARY_DEC type is a hybrid type with the integer portion encoded as binary
    % whilst the decimal portion is encoded as a greyscale
    BINARY_DEC = 2;  

    %GREY_SCALE type is a type that scales the value output between 1 and 0.
    GREY_SCALE = 3;


    %T_SCALE type; thermometor encoded values increase by one unit per neuron that fires.
    %These neurons fire in order (usually).
    T_SCALE = 4;
    Converted = [];
    for i = 1:size(Raw,2)
        if (type == BINARY)
            %Convert to rounded padded binery vector
            col = intToPBvec(Raw(:, i));          
         elseif (type == BINARY_DEC)
            %Convert to binery-decimal vector
            col = intToPBDvec(Raw(:, i));
         elseif (type == GREY_SCALE)
             %TODO
         elseif (type == T_SCALE) 
             %TODO
         end
         Converted = [Converted,col];
     end
      
end
