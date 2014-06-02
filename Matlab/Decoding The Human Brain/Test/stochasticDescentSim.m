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

fprintf('----- Attempting to Fine Tune the Network ----\n');
tic;
[ CNN, E_Hist ] = convStochGradientDescent(100, 1000, 'lms', CNN, double(yTrain), XCell, 0.0003, 0.0002, 0);
toc;