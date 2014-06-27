clear all;
disp('Fetching Training Data...\n');
path = '/Users/Karl/Projects/Dec2014Data/Data/';
subjects = 1:16;
[Xs, Ys] = loadData(subjects,path);
[X,Y] = concatData(Xs,Ys,subjects);

%Create Test Set and Permute X.
nExamples = size(X,1);
perm = randperm(nExamples);
fprintf('Permuting Dats\n');
fprintf('Perm X '); tic; X = X(perm,:); toc;
fprintf('Perm y '); tic; Y = Y(perm,:); toc;
 
nTraining = floor(nExamples * 9/10);
XTrain = X(1:nTraining,:);
yTrain = Y(1:nTraining,:);
XTest = X(nTraining+1:end,:);
yTest = Y(nTraining+1:end,:);

fprintf('------------------Training Neural Network------------------------\n');
[NN] = buildNeuralNetworkFromData(XTrain, double(yTrain), [40,10], 0, 'gen', [5,1]);
tic; [ NN, E_Hist] = stocGradientDescent(300, 800, 'lms', NN, double(yTrain), XTrain, 0.003, 0.002, 0); toc;
fprintf('------------------Classlification Accuracy-------------------\n');
Otr = predict(NN, XTrain);
Ots = predict(NN, XTest);
tacc = sum((Otr >= 0.5) == yTrain)/size(yTrain,1);
tstacc = sum((Ots >= 0.5) == yTest)/size(yTest,1);
fprintf('Training Accuracy \t %f\n', tacc);
fprintf('Test Accuracy \t %f\n', tstacc);

plot(E_Hist);
title('Neural Network on Pooled Data Error');
xlabel('Number of iterations');
ylabel('LMS Error');