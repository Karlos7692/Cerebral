% DecMeg2014 example code.
% Simple prediction of the class labels of the test set by:
%   - pooling all the triaining trials of all subjects in one dataset.
%   - Extracting the MEG data in the first 500ms from when the stimulus starts.
% - Using a linear classifier (elastic net).
% Implemented by Seyed Mostafa Kia (seyedmostafa.kia@unitn.it) and Emanuele
% Olivetti (olivetti@fbk.eu) as a benchmark for DecMeg 2014.
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
perm = randperm(nExamples);

fprintf('Permuting Dats\n');
fprintf('Perm X '); tic; X = X_train(perm,:); toc;
fprintf('Perm y '); tic; y = y_train(perm,:); toc;
 
nTraining = floor(nExamples * 9/10);
XTrain = X(1:nTraining,:);
yTrain = y(1:nTraining,:);
XTest = X(nTraining+1:end,:);
yTest = y(nTraining+1:end,:);

%Clear for memory purposes
fprintf('Clear Memory to save space\n');
clear XX; clear data; clear yy; clear X_train; clear y_train; clear X; clear features; clear filename;

%Change Type to match CNN Input.
XCell = cell(1,1);
XCell{1} = XTrain;

%Make Convolutional Neural Network.
[ CNN ] = generateCNN(infeats,inNNs);

fprintf('------------------Training Neural Network------------------------\n');
[ CNN, E_Collection ] = multistageTrain([100,0], 'stoc', 'lms', 800, CNN, double(yTrain), XCell, [0.03,0.003], [0.02,0.002], [0,0.1]);
fprintf('------------------Now Attempting to Fine Tune Network-----------------\n');
[ CNN, E_Hist] = convStochGradientDescent(200, 800, 'lms', CNN, double(yTrain), XCell, 0.03, 0.02, 0);
fprintf('Done!\n');
fprintf('------------------Classlification Accuracy-------------------\n');
Otr = convPredict(CNN, XCell);
TestCell = cell(1,1);
TestCell{1} = XTest;
Ots = convPredict(CNN, TestCell);
tacc = sum((Otr{1} > 0.5) == yTrain)/size(yTrain,1);
tstacc = sum((Ots{1} > 0.5) == yTest)/size(yTest,1);
fprintf('Training Accuracy \t %f\n', tacc);
fprintf('Test Accuracy \t %f\n', tstacc);
