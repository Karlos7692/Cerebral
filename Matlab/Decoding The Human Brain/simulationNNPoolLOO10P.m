clear all;
disp('DecMeg2014: https://www.kaggle.com/c/decoding-the-human-brain');
subjects_train = 1:16;    
disp(strcat('Training on subjects',num2str(subjects_train(1)),':',num2str(subjects_train(end))));
% We throw away all the MEG data outside the first 0.5sec from when
% the visual stimulus start:
tmin = 0;
tmax = 0.5;
disp(strcat('Restricting MEG data to the interval [',num2str(tmin),num2str(tmax),'] sec.'));

XsTrain = cell(16,1);
YsTrain = cell(16,1);
XsTest = [];
YsTest = [];
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
    nExamples = size(XX,1);
    nTraining = floor(nExamples * 9/10);
    XsTrain{i} = features(1:nTraining,:);
    XsTest = [XsTest;features(nTraining+1:end,:)];
    YsTrain{i} = yy(1:nTraining,:);
    YsTest = [YsTest;yy(nTraining+1:end,:)];
    
end



%Clear for memory purposes
fprintf('Clear Memory to save space\n');
clear XX; clear data; clear yy; clear X_train; clear y_train; clear X; clear features; clear filename;

%Creating Committee of Neural Networks
fprintf('Creating Committee of Neural Networls\n');
nNNs = 16;
Committee = cell(nNNs, 1);
weights = zeros(nNNs,1);
for i = 1:nNNs
    Committee{i} = buildNeuralNetworkFromData(XsTrain{1}, YsTrain{1}, [60,20], 0, 'gen', [5,1]);
    weights(i) = 1/nNNs;
end

%fprintf('Drawing Distributitions\n');
%tic; [ Distributions ] = bagging(XTrain, double(yTrain), nNNs); toc;

fprintf('Now Training the Committee of Neural Networks\n');
E_HistCollection = cell(16,1);
for i = 1:nNNs
    tic; [ NN, E_Hist] = gradientDescent(100, 'lms', Committee{i}, double(YsTrain{i}), XsTrain{i}, 0.003, 0.002, 0); toc;
    Committee{i} = NN;
    E_HistCollection{i} = E_Hist;
end




fprintf('Now Committee are making predictions\n');
[ OTs ] = committeePredict( Committee, XsTest, weights, 0.5); 
tstacc = sum((OTs > 0.5) == YsTest)/size(YsTest,1);
fprintf('Test Accuracy \t %f\n', tstacc);    

