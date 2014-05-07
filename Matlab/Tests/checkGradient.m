function [diff, J, grad] = checkGradient(NN, Y, X, lambda)

    X = maintainNN(NN,X);
    [J, grad] = costFunc(NN, Y, X, lambda);
    %Minus gradient, already added.
    approx = approxGradient(NN, X, Y, lambda);
    

    %Manual Gradient checking.
    %disp([approx grad]);
    %fprintf(['The above two columns you get should be very similar.\n' ...
    %     '(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n']);
    
    
   
    % Evaluate the norm of the difference between two solutions.  
    % If you have a correct implementation, and assuming you used EPSILON = 0.0001 
    % in computeNumericalGradient.m, then diff below should be less than 1e-9
    diff = norm(approx-grad)/norm(approx+grad);
    
    fprintf(['If your backpropagation implementation is correct, then \n' ...
         'the relative difference will be small (less than 1e-9). \n' ...
         '\nRelative Difference: %g\n'], diff);
end



