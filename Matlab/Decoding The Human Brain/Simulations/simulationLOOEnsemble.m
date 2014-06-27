
%SIMULATIONLOOENSEMBLE Summary of this function goes here
%   Detailed explanation goes here
disp('Fetching Training Data...\n');
path = '/Users/Karl/Projects/Dec2014Data/Data/';
subjects = 1:15;
nChannels = 306;
nFeats = 125;

[ Xs, Ys ] = loadData( subjects, path );

fprintf('Concatinating Data... \n');
tic; [ X,Y ] = concatData( Xs, Ys, subjects); toc;
clear Xs; clear Ys;

fprintf('Permuting Data...\n');
tic;
perm = randperm(size(X,1));
X = X(perm,:);
Y = Y(perm,:);
toc;

fprintf('Fethcing CV Data...\n');
testsubject = 16;
[ XCV, YCV ] = loadData( testsubject, path );

[ ComponentMatrix ] = generateComponentMatrix(nFeats, nChannels);

%transform requires type cell.
XTemp = cell(1,1); 
XTemp{1} = X;

%Part of Core Convolutional Functions. to split the input space into
%feature spaces.
[ XTraining ] = transform(ComponentMatrix, XTemp); 
[ XCV ] = transform(ComponentMatrix, XCV); 
clear XTemp;


fprintf('Making Ensemble of Networks..\n');
Ensemble = cell(nChannels,1);
tic;
for i = 1:nChannels
    [NN] = buildNeuralNetworkFromData(XTraining{1}, double(Y), 30, 0, 'gen', [5,1]);
    Ensemble{i} = NN;
end
toc;

fprintf('Training All Networks...\n');
E_History = [];
for i = 1:nChannels
    tic; [ NN, E_Hist] = stocGradientDescent(100, 550, 'lms', Ensemble{i}, double(Y), XTraining{i}, 0.001, 0.0003, 0); toc;
    E_History = [E_History, E_Hist];
end


fprintf('Getting Predictions...\n');
weights = ones(nChannels,1) * 1/nChannels;
tic; OTr = committeePredictDistributions( Ensemble, XTraining, weights) >= 0.5; toc;
tic; OTs = committeePredictDistributions( Ensemble, XCV, weights) >= 0.5; toc;

fprintf('Output Accuracy and Plots...\n');
tacc = sum(OTr == logical(Y))/size(Y,1);
tstacc = sum(OTs == logical(YCV{1}))/size(YCV{1},1);

fprintf('Training Accuracy: %f\n', tacc);
fprintf('Test Accuracy: %f\n', tstacc);

plot(E_History);
title('Ensemble Network Experiment: Error over Time');
xlabel('Number of Iterations');
ylabel('LMS Error');


