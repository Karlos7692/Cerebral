

function bdvec = intToPBDvec(realvec)
    PADDING = 1;
    mvl = 0;
    bdvec = [];
    intvec = fix(realvec);
    decvec = realvec - intvec;
    
    %Padding values
    value = max(intvec);
    mvl = length(intToBvec(value));
    
    for i = 1:length(intvec)
        bvTemp = intToBvec(intvec(i));
        dec = decvec(i);
        diff = mvl - length(bvTemp);
        bvTemp = [(zeros(1,diff)), bvTemp];
        bdvTemp = [bvTemp, dec];    
        bdvec = [bdvec; bdvTemp];
    end      
end
