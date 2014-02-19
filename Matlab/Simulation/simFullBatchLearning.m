function [Pred, PDiff, Corr, Psd] = simFullBatchLearning(NNParams, LearningParams, TDataParams);
%SIMBATCH Summary of this function goes here
%   Detailed explanation goes here

%Unpack Training Data Parameters and build
[RawIn, RawOut, CVIn, CVOut] = prepareRawData(TDataParams.Data, TDataParams.predict, TDataParams.cvsize);


%Copy Data Values for next day prediction
RawInTemp = RawIn;
RawOutTemp = RawOut;
CVInTemp = CVIn;
CVOutTemp = CVOut;


%Initialise 
DroppedRawIn = [];
DroppedRawOut = [];
Pred = zeros(size(CVOut));


%Should be one value we know in the future
fprintf('************ Simulation Begin ***************\n'); 

for i = 1:length(CVOut)-1

    %Unpack Parameters and build
    fprintf('Neural Network Weights\n');
    [NN TrainData TargData] = buildNeuralNetwork(RawInTemp, RawOutTemp, NNParams.hidden, NNParams.nStateVecs, NNParams.type, NNParams.encoding);
    [NN, J_Hist] = gradientDescent(LearningParams.maxIter, NN, TargData, TrainData, LearningParams.lr, LearningParams.momentum, LearningParams.reg);
    
    %Analyse Training set Fit

    [dummy,thresh] = analyseTSet(J_Hist, NN, TrainData, RawOut, NNParams.threshType);
    
    %Predict for next day
    Input = [RawInTemp ; CVInTemp(1, :)]; 
    %Make Input Temoral
    Input = prepareTNN(NN, Input);
    %Get last feature values
    Input = Input(end,:);
    %Get Prediction
    Pred(i) = predict(NN, Input, NNParams.threshType, thresh);
    
    %Print the prediction values, reals values and difference
    fprintf('                 Day,   Real,    Pred,    Diff\n '); 
    fprintf('Prediction Day: %d,    %f, %f, %f\n', i, CVOut(i), Pred(i), (CVOut(i) - Pred(i))); 
    
    %Rollover 1 day.
    [RawInTemp, RawOutTemp, CVInTemp, CVOutTemp, DroppedRawIn, DroppedRawOut] = rollover(RawIn, RawOut, CVInTemp, CVOutTemp, DroppedRawIn, DroppedRawOut);
    
    
end
[stats] = analyseCVSim(Pred, CVOut);
PDiff = CVOut - Pred;
Corr = sqrt(stats.rsquare);
Psd = std(CVOut - Pred);



end

