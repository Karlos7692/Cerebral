%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% Title: Raw Data Converter                                                                                    %
%                                                                                                              %
% Author: Karl Nelson                                                                                          %
% Email: <k.c.nelson7692@gmail.com>                                                                            %
%                                                                                                              %
% Description:                                                                                                 %
%                                                                                                              %
% Converts Raw Data into vectorised format for specific Neural Networks                                        %
%                                                                                                              %
% Parameters:                                                                                                  %
%                                                                                                              %
% NN: A Neural Network                                                                                         %
% Raw: Raw Data in Matrix Form                                                                                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function TData = convertRaw(NN, Raw)
      if strcmp(NN.TEMPORAL, NN.type)
          TData = prepareTDataTNN(NN, Raw);   
      end
end




function TData = prepareTDataTNN(NN, Raw)

    nStateUnits = sum(NN.state);
    statePad = zeros(size(Raw,1), nStateUnits);
          
    %fsui - First state unit index
    fsui = size(Raw,2) + 1;
         
    TData = [Raw, statePad];
          
    firstSVec = 1;
    for i = 1:(size(Raw,1) - 1)
         stateW = Raw(i:-1:firstSVec, :);
              
         %For readbility, the state Weight Matrix (stateW)
         %is transposed to aline the state vector- inputs correctly
         stateW = stateW';
              
              
         stateVec = stateW(:);
         nZeroedState = nStateUnits - length(stateVec);
         stateVec = stateVec';
              
         if nZeroedState <= 0
             firstSVec = firstSVec + 1;
         else;   
              stateVec = [stateVec, zeros(1,nZeroedState)];
         end
              
          TData((i+1),fsui:end) = TData((i+1),fsui:end) + stateVec;
    end  

emd
