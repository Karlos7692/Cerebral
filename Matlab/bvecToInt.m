





function i = bvecToInt(bvec)
     degree = length(bvec) -1;
     
     power = degree:-1:0;
     base = ones(1,length(bvec)).*2;
     i = sum(bvec.*base.^power);
end
