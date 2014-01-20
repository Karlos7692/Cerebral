



function real = bdvecToReal(bdvec)
    i = bvecToInt(bdvec(1:(end-1)));
    real = i + bdvec(end);
end
