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
Xs = cell(15,1);
Ys = cell(15,1);
X_test = [];
y_test = [];
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
    Xs{i} = features;
    Ys{i} = double(yy);
end


fprintf('Calculating Covariate Shift...\n');
nNNs = 16;
NNs = cell(nNNs,1);
alphas = ones(nNNs,1) * 0.003;
mews = ones(nNNs,1) * 0.002;
lambdas = zeros(nNNs,1);
for i = 1:nNNs
  NNs{i} = buildNeuralNetworkFromData(Xs{i},  Ys{i}, [40,10], 0, 'gen', [5,1]);
end    
DistWeights = simpleCovariateShift( Xs, Ys, NNs, alphas, mews, lambdas);
fprintf('Combining Distribution Weights to match Training Set...\n');
distweights = [];
for i = 1:nNNs;
    fprintf('Subject %d: Ptest/Ptrain = %f\n', i, DistWeights{i}(1));
    distweights = [distweights;DistWeights{i}];
end
clear nNNs; clear NNs; clear Xs; clear Ys;

fprintf('------------------Training Neural Network------------------------\n');
[NN] = buildNeuralNetworkFromData(X_train, double(y_train), [40,10], 0, 'gen', [5,1]);
tic; [ NN, E_Hist] = weightedStocGradientDescent(100, 600, distweights, 'lms', NN, double(y_train), X_train, 0.003, 0.002, 0); toc;
fprintf('------------------Classlification Accuracy-------------------\n');
Otr = predict(NN, X_train);
tacc = sum((Otr >= 0.5) == y_train)/size(y_train,1);
fprintf('Training Accuracy \t %f\n', tacc);
