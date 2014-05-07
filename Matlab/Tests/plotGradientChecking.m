function [ relDiffs, accuracy] = plotGradientChecking( NN, Y, X, lambda )
%PLOTGRADIENTCHECKING Summary of this function goes here
%   Detailed explanation goes here
    
    accuracy = 3.5:0.01:5.5;
    relDiffs = zeros(size(accuracy));
    Approx = [];
    Grad = [];
    i = 1;
    for acc = accuracy
        X = maintainNN(NN,X);
        [J, grad] = costFunc(NN, Y, X, lambda);
        %Minus gradient, already added.
        approx = adjustApproxGradient(NN, X, Y, lambda,acc);
    
        
        figure;
       % plot(1:length(approx), approx, 'r');
        hold on
       % plot(1:length(grad), grad, 'g');
        
        plot3((1:length(approx))',approx,grad);
        pause;
        %title(strcat('Using acc: ',num2str(acc)));
        %Manual Gradient checking.
       %disp([approx grad]);
       %fprintf(['The above two columns you get should be very similar.\n' ...
        %     '(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n']);
    
    
   
        % Evaluate the norm of the difference between two solutions.  
        % If you have a correct implementation, and assuming you used EPSILON = 0.0001 
        % in computeNumericalGradient.m, then diff below should be less than 1e-9
        diff = norm(approx-grad)/norm(approx+grad);
        relDiffs(i) = diff;
        i=i+1;
    end
    hold off
    plot(accuracy, relDiffs);
    %xlabel('Acuracy

end

