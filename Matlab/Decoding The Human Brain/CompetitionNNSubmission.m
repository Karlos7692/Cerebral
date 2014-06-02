
function [filename_submission, NN] = CompetitionNNSubmission(predictorname)
    
    fprintf('Loading Neural Network\n');
    Wks = load(strcat('/Users/Karl/Projects/DecodingHumanBrainSubmissions/', predictorname));
    NN = Wks.NN;
    
    
    tmin = 0;
    tmax = 0.5;
    X_test = [];
    ids_test = [];
    disp('Creating the testset.');
    subjects_test = 17:23;
    for i = 1 : length(subjects_test)
        path = '/Users/Karl/Projects/Dec2014Data/Test/'; % Specify absolute path
        filename = sprintf(strcat(path,'test_subject%02d.mat'),subjects_test(i));
        disp(strcat('Loading ',filename));
        data = load(filename);
        XX = data.X;
        ids = data.Id;
        sfreq = data.sfreq;
        tmin_original = data.tmin;
        disp('Dataset summary:')
        disp(sprintf('XX: %d trials, %d channels, %d timepoints',size(XX,1),size(XX,2),size(XX,3)));
        disp(sprintf('Ids: %d trials',size(ids,1)));
        disp(strcat('sfreq:', num2str(sfreq)));
        features = createFeatures(XX,tmin, tmax, sfreq,tmin_original);
        X_test = [X_test;features];
        ids_test = [ids_test;ids];
    end
    
    Out = predict(NN, X_test)
    
    Res = Out > 0.8;
    
    
    filename_submission = 'submission.csv';
    disp(strcat('Creating submission file: ',filename_submission));
    f = fopen(filename_submission, 'w');
    fprintf(f,'%s,%s\n','Id','Prediction');
    for i = 1 : length(Res)
        fprintf(f,'%d,%d\n',ids_test(i),Res(i));
    end
    fclose(f);
    disp('Done.');
    
end