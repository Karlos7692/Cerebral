clear all;

disp('Fetching Training Data...\n');
path = '/Users/Karl/Projects/Dec2014Data/Data/';
subjects = 1:16;
[Xs, Ys] = loadData(subjects,path);
[ X,Y ] = concatData( Xs, Ys, subjects(1:15));

%Create Test Set and Permute X.
nExamples = size(X,1);
perm = randperm(nExamples);
XTrain = X(perm,:);
YTrain = Y(perm,:);
XTest = Xs{16};
YTest = Ys{16};
%[ Distribution ] = bagging(X_train, double(y_train), 1, 5);
%clear X_train; clear y_train;
 


fprintf('------------------Training Neural Network------------------------\n');
[NN] = buildNeuralNetworkFromData(XTrain, double(YTrain), [40,10], 0, 'gen', [5,1]);
tic; [ NN, E_Hist] = stocGradientDescent(400, 580, 'lms', NN, double(YTrain), XTrain, 0.001, 0.0003, 0.2); toc;
fprintf('------------------Classlification Accuracy-------------------\n');
Otr = predict(NN, XTrain);
Ots = predict(NN, XTest);
tacc = sum(((Otr >= 0.5) == logical(YTrain)))/size(YTrain,1);
tstacc = sum(((Ots >= 0.5) == logical(YTest)))/size(YTest,1);
fprintf('Training Accuracy \t %f\n', tacc);
fprintf('Test Accuracy \t %f\n', tstacc);


plot(E_Hist);
title('Neural Network on Pooled Data Error');
xlabel('Number of iterations');
ylabel('LMS Error');