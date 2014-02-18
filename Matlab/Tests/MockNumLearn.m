function [J10, J100] = MockNumLearn(nntype, nStatevecs, mom)

if(strcmp(nntype, 'gen') || isempty(nStatevecs))
    nStatevecs = 0;
end


  
    RawIn = (1:300)';
    RawOut = (2:301)';
    
    [NN1, TrainData, TargData] = buildNeuralNetwork(RawIn, RawOut, 10, nStatevecs, nntype, [1,1]);
    NN2 = buildNeuralNetwork(RawIn, RawOut, 100, nStatevecs, nntype, [1,1]);
    
   options = optimset('MaxIter', 400);
   costFunction1 = @(p) lmsTest( p, ...
               NN1.shape(1), ...
               NN1.shape(2), ...
               NN1.shape(3), ...
               TrainData, TargData, 0);
    
           
  [nn_params, cost] = fmincg(costFunction1, NN1.weights, options);
  pause;
  
  costFunction2 = @(p) lmsTest( p, ...
               NN2.shape(1), ...
               NN2.shape(2), ...
               NN2.shape(3), ...
               TrainData, TargData, 0);
           
  [nn_params, cost] = fmincg(costFunction2, NN2.weights, options);
  %  pause;
  %  maxIter = 40;
  %  for i = 1:maxIter
  %      figure;
  %      [NN, TrainData, TargData] = buildNeuralNetwork(RawIn, RawOut, 10, nStatevecs, nntype, [1,1]);
  %      [NN, J_Hist] = gradientDescent(12000, NN, TargData, TrainData, (i*0.0002 + 0.001), mom, 0);
  %      hold on
  %      J10(i) = J_Hist(end);
  %      plot(1:length(J_Hist), J_Hist, 'b');
  %      [NN, TrainData, TargData] = buildNeuralNetwork(RawIn, RawOut, 100, nStatevecs, nntype, [1,1]);
  %      [NN, J_Hist] = gradientDescent(12000, NN, TargData, TrainData, (i*0.00002 + 0.0005), mom, 0);
  %      J100(i) = J_Hist(end);
  %      plot(1:length(J_Hist), J_Hist, 'r');
  %      legend('10 HUs', '100 HUs');
  %      xlabel('Epochs');
  %      ylabel('LMS Error');
  %      title(strcat('10 HUs vs 100 HUs, LR-10 = ', num2str(i*0.0002 + 0.001), ' LR-100 = ', num2str(i*0.00002 + 0.0005)));
  %      
  %  end
  %  figure
  %  J10LR = (0.0002 + 0.001):0.0002:(maxIter*0.0002 +0.001);
  %  length(J10LR)
  %  length(J10)
  %  J100LR = (0.00002 + 0.0005):0.00005:(maxIter*0.00002 + 0.0005);
  %  xlabel('Learning Rate');
  %  ylabel('LMS Error');
  % title('Cost per Learning Rate for 10/100 HUs');
  %  plot(J10LR, J10, 'b', 'Marker', '+');
  %  hold on
  %  plot(J100LR, J100, 'g', 'Marker', '+');
  % legend('10 HUs', '100 HUs');
  % hold off
    
    
end