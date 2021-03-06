%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% Title: Data Reader                                                                                           %
%                                                                                                              %
% Author: Karl Nelson                                                                                          %
% Email: <k.c.nelson7692@gmail.com>                                                                            %
%                                                                                                              %
% Description:                                                                                                 %
%                                                                                                              %
% Reads and sanitises imported files for learning machine training data.                                       %
%                                                                                                              %
%                                                                                                              %
%                                                                                                              %
% Parameters:                                                                                                  %
%                                                                                                              %
% filepath: (string) filepath to file to import                                                                %
%                                                                                                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [TData] = readData(filepath)
     % Control
     [pathstr, name, ext] = fileparts(filepath);
     
     % Import csv data
     if (strcmp('.csv', ext))
         %Dates 1st column
         %Titles 1st row
         TData = csvread(filepath);
         TData = TData(2:end, 2:end);
     end
end

