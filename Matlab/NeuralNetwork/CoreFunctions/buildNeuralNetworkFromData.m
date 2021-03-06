%Encoding Binary: 1
%Encoding Binary Decimal: 2


function [NN TrainData TargData] = buildNeuralNetworkFromData(RawIn, RawOut, hidden, nStateVecs, type, encoding)
     
     %TODO change prepareTNN nicer code.
     
     BINARY = 1;
     BINARYDEC = 2;
     
     GENERAL = 'gen';
     TEMPORAL = 'tem';
     SRN = 'srn';
          
     inFeats = size(RawIn, 2);
     outFeats = size(RawOut,2);
     
     inenc = encoding(1);
     outenc = encoding(2);
     
     
     
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     %              Input Shape and Conversions              %
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     
     %Convert input features to input shape
     %TrainData - Training Data for generic neural network
     
     TrainData = [];
     if (inenc == BINARY)
         TrainData = convertData(RawIn, BINARY);
     elseif (inenc == BINARYDEC)
         TrainData = convertData(RawIn, BINARYDEC);
     else
         TrainData = RawIn;
     end
     
     nInUs = size(TrainData,2);
     nHUs = hidden(1);
     
     %Getting input shape according to neural network type
     nStateUs = 0;
     if (strcmp(type, TEMPORAL))
         nStateUs = nInUs * nStateVecs;
     elseif(strcmp(type, SRN))
         nStateUs = nHUs * nStateVecs;
     end
     
     %Number of input units for the input shape
     inShape = nInUs + nStateUs;
     
     
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     %            Hidden Unit Shape Conversions              %
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     
     hiddenShape = hidden;
     
     
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     %             Output Shape and Conversions              %
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     
     TargData = [];
     output = [];
     for i = 1:size(RawOut,2);
         if (outenc == BINARY)
             TargTemp = convertData(RawOut(:, i), BINARY);
         elseif (outenc == BINARYDEC)
             TargTemp = convertData(RawOut(:, i), BINARYDEC);
         else
             TargTemp = RawOut(:,i);
         end
         TargData = [TargData, TargTemp];
         output = [output, size(TargTemp, 2)];
     end
     
     outShape = sum(output);
     
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     %                 Neural Network Shape                  %
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     
     shape = [inShape, hiddenShape, outShape];
     
     
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     %                 State Unit Shape                      %
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     
     if (strcmp(type, GENERAL))
         state = [];
     elseif (strcmp(type, TEMPORAL))
         state = ones(1,nStateVecs).*nInUs;
     elseif (strcmp(type,SRN))
         state = ones(1,nStateVecs).*nHUs;
     else 
         state = [];
     end
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     %                Neural Network Seeding                 %
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     
     %genetates a seed vector from nerual network shape
     seedvec = generateSeedvec(shape);
     
     
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     %                Build Neural Network                   %
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     NN = NeuralNetwork(shape, state, output, outenc, type, seedvec);
     
     
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     %                Build Training Data                    %
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     if(strcmp(NN.type, TEMPORAL))
         TrainData = prepareTNN(NN,TrainData);
     elseif(strcmp(NN.type, SRN))
         TrainData = prepareSRN(NN,TrainData);
     else
         
     end
     size(NN.weights)
end