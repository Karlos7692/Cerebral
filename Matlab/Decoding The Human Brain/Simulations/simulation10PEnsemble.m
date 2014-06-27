
clear all;
disp('Fetching Training Data...\n');
path = '/Users/Karl/Projects/Dec2014Data/Data/';
subjects = 1:16;
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

fprintf('Making Training and Cross Validation Set...\n');
tic;
m = floor(9/10 * size(X,1));
XTrain = X(1:m,:);
YTrain = Y(1:m,:);
XCV = X(m+1:end,:);
YCV = Y(m+1:end,:);
toc;

fprintf('Transforming Input Space of all Channels to one Channel per Neural Network...\n');
tic;

%Used to determine where all the features go. In this case features are not
%reused.
[ ComponentMatrix ] = generateComponentMatrix(nFeats, nChannels);

%transform requires type cell.
XTemp = cell(1,1); 
XTemp{1} = XTrain;

%Part of Core Convolutional Functions. to split the input space into
%feature spaces.
[ XTraining ] = transform(ComponentMatrix, XTemp); 
clear XTemp;

%Do the same for XCV
XTemp = cell(1,1);
XTemp{1} = XCV;

[ XCVs ] = transform(ComponentMatrix, XTemp);
toc;

fprintf('Making Ensemble of Networks..\n');
Ensemble = cell(nChannels,1);
tic;
for i = 1:nChannels
    [NN] = buildNeuralNetworkFromData(XTraining{1}, double(YTrain), 30, 0, 'gen', [5,1]);
    Ensemble{i} = NN;
end
toc;

fprintf('Training All Networks...\n');
E_History = [];
for i = 1:nChannels
    fprintf('NN :%d ||',i);
    tic; [ NN, E_Hist] = stocGradientDescent(400, 800, 'lms', Ensemble{i}, double(YTrain), XTraining{i}, 0.03, 0.02, 0); toc;
    E_History = [E_History, E_Hist];
end

fprintf('Getting Predictions...\n');
weights = ones(nChannels,1) * 1/nChannels;
tic; OTr = committeePredictDistributions( Ensemble, XTraining, weights) >= 0.5; toc;
tic; OTs = committeePredictDistributions( Ensemble, XCVs, weights) >= 0.5; toc;

fprintf('Output Accuracy and Plots...\n');
tacc = sum(OTr == YTrain)/size(YTrain,1);
tstacc = sum(OTs == YCV)/size(YCV,1);

fprintf('Training Accuracy: %f\n', tacc);
fprintf('Test Accuracy: %f\n', tstacc);

plot(E_History);
title('Ensemble Network Experiment: Error over Time');
xlabel('Number of Iterations');
ylabel('LMS Error');
