








function seedvec = generateSeedvec(vec)
     l = length(vec) - 1;
     seedvec = [];
     for i = 1:l
         s = rand('seed');
         seedvec = [s, seedvec];
         rand();
     end

end
