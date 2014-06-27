function [ Xs, Ys ] = loadData( subjects, path )
%LOADDATA Summary of this function goes here
%   Detailed explanation goes here
    disp('DecMeg2014: https://www.kaggle.com/c/decoding-the-human-brain');
        
    disp(strcat('Training on subjects',num2str(subjects(1)),':',num2str(subjects(end))));
    % We throw away all the MEG data outside the first 0.5sec from when
    % the visual stimulus start:
    tmin = 0;
    tmax = 0.5;
    
    % Crating the trainset. (Please specify the absolute path for the train data) 
    Xs = cell(numel(subjects),1);
    Ys = cell(numel(subjects),1);
    disp('Creating the trainset.');
    for i = 1 : length(subjects)

        filename = sprintf(strcat(path,'train_subject%02d.mat'),subjects(i));
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
        Xs{i} = features;
        Ys{i} = logical(yy);
    end

end

