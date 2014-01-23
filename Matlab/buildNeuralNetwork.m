






%Encoding Binary: 1
%Encoding Binary Decimal: 2


function [NN TrainData TargData] = buildNeuralNetwork(RawIn, RawOut, hidden, nStateVecs, type, encoding)
     
     %TODO change prepareTDataTNN nicer code.
     BINARY = 1;
     BINARYDEC = 2;
     
     TEMPORAL = 'tem';
     
     inFeats = size(RawIn, 2);
     outFeats = size(RawOut,2);
     
     inenc = encoding(1);
     outenc = encoding(2);
     
     
     
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     %              Input Shape and Convertions              %
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
     
     %Getting input shape according to neural network type
     nStateUs = 0;
     if (strcmp(type, TEMPORAL))
         nStateUs = nInUs * nStateVecs;
     end
     
     %Number of input units for the input shape
     inShape = nInUs + nStateUs;
     
     
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     %            Hidden Unit Shape Conversions              %
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     
     hiddenShape = hidden;
     
     
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     %             Output Shape and Convertions              %
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
     state = ones(1,nStateVecs).*nInUs;
     
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
     
     TrainData = prepareTDataTNN(NN,TrainData);
     size(NN.weights)
end
