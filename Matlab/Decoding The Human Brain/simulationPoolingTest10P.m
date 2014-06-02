clear all;
disp('DecMeg2014: https://www.kaggle.com/c/decoding-the-human-brain');
subjects_train = 1:16;    
disp(strcat('Training on subjects',num2str(subjects_train(1)),':',num2str(subjects_train(end))));
% We throw away all the MEG data outside the first 0.5sec from when
% the visual stimulus start:
tmin = 0;
tmax = 0.5;
disp(strcat('Restricting MEG data to the interval [',num2str(tmin),num2str(tmax),'] sec.'));
X_train = [];
y_train = [];
X_test = [];
ids_test = [];
% Crating the trainset. (Please specify the absolute path for the train data)
inNNs = 306;
infeats = 125;
disp('Creating the trainset.');
for i = 1 : length(subjects_train)
    path = '/Users/Karl/Projects/Dec2014Data/Data/';  % Specify absolute path
    filename = sprintf(strcat(path,'train_subject%02d.mat'),subjects_train(i));
    disp(strcat('Loading ',filename));
    data = load(filename);
    XX = data.X;
    yy = data.y;
    
    sfreq = data.sfreq;
    tmin_original = data.tmin;
    disp('Dataset summary:')
    disp(sprintf('XX: %d trials, %d channels, %d timepoints',size(XX,1),size(XX,2),size(XX,3)));
    disp(sprintf('yy: %d trials',size(yy,1)));
    disp(strcat('sfreq:', num2str(sfreq)));
    [features] = createFeatures(XX,tmin, tmax, sfreq,tmin_original);
    X_train = [X_train;features];
    y_train = [y_train;yy];
end


%Create Test Set and Permute X.
nExamples = floor(size(X_train,1));

nTraining = floor(nExamples * 9/10);
perm = randperm(nTraining);
XTrain = X_train(perm,:);
yTrain = y_train(perm,:);
XTest = X_train(nTraining+1:end,:);
yTest = y_train(nTraining+1:end,:);

%Clear for memory purposes
fprintf('Clear Memory to save space\n');
clear XX; clear data; clear yy; clear X_train; clear y_train; clear X; clear features; clear filename;


%Train Committee
fprintf('----- Training Committee Machine -----\n');
Committee = cell(inNNs,1);
[ ComponentMatrix ] = generateComponentMatrix(infeats, inNNs);
Xs = cell(1,1);
Xs{1} = XTrain;
[ XSplits ] = transform(ComponentMatrix, Xs);
for i = 1:inNNs
   
    [NN, TrainData, TargData] = buildNeuralNetworkFromData(XSplits{i}, double(yTrain), 30, 0, 'gen', [5,1]);
    tic; [ NN, E_Hist] = stocGradientDescent(200, 800, 'lms', NN, TargData, TrainData, 0.3, 0.2, 0); toc;
    fprintf('- NN Error - %e\n', E_Hist(end));
    plot(E_Hist);
    Committee{i} = NN;
end



fprintf('--- Committee Voting on Prediction ---\n');
XsTrain = cell(1,1);
XsTest = cell(1,1);
XsTrain{1} = XTrain;
XsTest{1} = XTest;
[ XSplitsTrain ] = transform(ComponentMatrix,XsTrain);
[ XSplitsTest ] = transform(ComponentMatrix, XsTest);
%Predict Voting
TrainingResults = zeros(size(yTrain,1), inNNs);
TestResults = zeros(size(yTest,1), inNNs);
for i = 1:inNNs
    TrainingResults(:,i) = (predict(Committee{i}, XSplitsTrain{i}) >= 0.5);
    TestResults(:,i) = (predict(Committee{i}, XSplitsTest{i}) >= 0.5);
end

pooledTrainingResult = (1/inNNs * sum(TrainingResults,2)) >= 0.5;
pooledTestResult = (1/inNNs * sum(TestResults,2)) >= 0.5;

tacc = sum(pooledTrainingResult == yTrain)/size(yTrain,1);
fprintf('Training Accuracy \t %f\n', tacc);
tstacc = sum(pooledTestResult == yTest)/size(yTest,1);
fprintf('Test Accuracy \t %f\n', tstacc);



