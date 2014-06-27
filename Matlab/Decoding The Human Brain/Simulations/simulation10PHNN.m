% DecMeg2014 example code.
% Simple prediction of the class labels of the test set by:
%   - pooling all the triaining trials of all subjects in one dataset.
%   - Extracting the MEG data in the first 500ms from when the stimulus starts.
% - Using a linear classifier (elastic net).
% Implemented by Seyed Mostafa Kia (seyedmostafa.kia@unitn.it) and Emanuele
% Olivetti (olivetti@fbk.eu) as a benchmark for DecMeg 2014.
clear all;
path = '/Users/Karl/Projects/Dec2014Data/Data/';
subjects = 1:16;
inFeats = 125;
inNNs = 306;
tic;
fprintf('Fetching Data...\n');
[Xs, Ys] = loadData(subjects,path);
[X, Y] = concatData( Xs, Ys,subjects);
toc;
%Create Test Set and Permute X.
nExamples = size(X,1);
perm = randperm(nExamples);
fprintf('Permuting Dats\n');
fprintf('Perm X '); tic; X = X(perm,:); toc;
fprintf('Perm y '); tic; Y = Y(perm,:); toc;
 
nTraining = floor(nExamples * 7/10);
stacking = nTraining + floor(2/10 * nExamples); 

XTrain = X(1:nTraining,:);
yTrain = Y(1:nTraining,:);

XStack = X(nTraining+1:stacking,:);
YStack = Y(nTraining+1:stacking,:);
XStackCell = cell(1,1);
XStackCell{1} = XStack;

XTest = X(stacking+1:end,:);
yTest = Y(stacking+1:end,:);

%Change Type to match CNN Input.
XCell = cell(1,1);
XCell{1} = XTrain;

%Make Convolutional Neural Network.
[ CNN ] = generateCNN(inFeats,inNNs, 30,10);

fprintf('------------------Training Neural Network------------------------\n');
[ CNN, E_Collection ] = multistageTrain([200,1], 'stoc', 'lms', 800, CNN, double(yTrain), XCell, [0.03,0.003], [0.02,0.002], [0,0.1]);
fprintf('------------------Now Attempting to Fine Tune Network-----------------\n');
[ CNN, E_Hist] = convStochGradientDescent(30, 800, 'lms', CNN, double(YStack), XStackCell, 0.0003, 0.0002, 0);
fprintf('Done!\n');
fprintf('------------------Classlification Accuracy-------------------\n');
%Join X training and stacking 
XCell{1} = [XCell{1};XStackCell{1}];
Otr = convPredict(CNN, XCell);
TestCell = cell(1,1);
TestCell{1} = XTest;
Ots = convPredict(CNN, TestCell);
tacc = sum((Otr{1} >= 0.5) == logical(Y(1:stacking,:)))/size(Y(1:stacking,:),1);
tstacc = sum((Ots{1} >= 0.5) == logical(yTest))/size(yTest,1);
fprintf('Training Accuracy \t %f\n', tacc);
fprintf('Test Accuracy \t %f\n', tstacc);


figure;
plot(E_Collection{1});
title('1st Tier Network Errors');
xlabel('Number of Iterations');
ylabel('LMS Error');

figure;
plot(E_Collection{2})
title('2nd Tier Network');
xlabel('Number of Iterations');
ylabel('LMS Error');

figure;
plot(E_Hist);
title('Network Error after Fine Tuning');
xlabel('Number of Iterations');
ylabel('LMS Error');
