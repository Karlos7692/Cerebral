



function bvec = intToPBvec(intvec)
    PADDING = 1;
    mvl = 0;
    bvec = [];
    value = max(intvec);
    mvl = length(intToBvec(value));
        
    for i = 1:length(intvec)
        bvTemp = intToBvec(intvec(i));
        diff = mvl - length(bvTemp);
        bvTemp = [(zeros(1,diff)), bvTemp];    
        bvec = [bvec; bvTemp];
    end      
end
