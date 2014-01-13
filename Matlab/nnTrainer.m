%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% Title: Neural Network Trainer                                                                                %
%                                                                                                              %
% Author: Karl Nelson                                                                                          %
% Email: <k.c.nelson7692@gmail.com>                                                                            %
%                                                                                                              %
% Description:                                                                                                 %
%                                                                                                              %
%Trains specific Neural Networks according to their structure.                                                 %
%                                                                                                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [TrainedNN] = nnTrainer (NN, TData)
      if strcmp(NN.type , "SRN")
      
      elsif strcmp (NN.type ,"Temporal")
          % Setup Training Data from raw data
          % Cost Propogation
          % Backpropogation
          %    - Gradient Descent: + Regularisation
          %    - Gradient Checking
      end
end
