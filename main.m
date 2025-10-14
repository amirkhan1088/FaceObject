% Download store the databases in the local memory in the same folder and
% change the path of rootFolder accordingly the final folder name is
% already mentioned
% Run for one database at a time
%% GIT face database Link
%https://www.anefian.com/research/face_reco.htm
categories = {'s01','s02','s03','s04','s05','s06','s07','s08','s09','s10',...
    's11','s12','s13','s14','s15','s16','s17','s18','s19','s20',...
    's21','s22','s23','s24','s25','s26','s27','s28','s29','s30',...
    's31','s32','s33','s34','s35','s36','s37','s38','s39','s40',...
    's41','s42','s43','s44','s45','s46','s47','s48','s49','s50'};

rootFolder = 'gt_db';
imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource',...
    'foldernames');  % create the imagestore from the dataset

% split imds into training and test sets
[Train Test] = splitEachLabel(imds,0.55,'randomized');
% This ratio of 0.55 gives 8 images in train set and 7 images in test set
% Number of images in each class can be checked by following command
countEachLabel(Train);
countEachLabel(Test);
Train = shuffle(Train); % To shuffle the Training set
Test = shuffle(Test);   % To shuffle the Test set

L = length(Train.Labels); % length of training set
L1 = length(Test.Labels); % length of test set
Y = Train.Labels; % Labels of training set
Y1 = Test.Labels; % Labels of test set


%% Extended Yale B database download link below
% http://vision.ucsd.edu/datasets/extended-yale-face-database-b-b

% create the folder ExtYaleB and store the dataset there
rootFolder = 'ExtYaleB'; % without ambient image files 

categories = {'yaleB11','yaleB12', 'yaleB13', 'yaleB15','yaleB16',...
    'yaleB17','yaleB18', 'yaleB19', 'yaleB20','yaleB21', 'yaleB22', 'yaleB23',...
    'yaleB24','yaleB25','yaleB26','yaleB27', 'yaleB28', 'yaleB29','yaleB30',...
    'yaleB31','yaleB32', 'yaleB33', 'yaleB34', 'yaleB35','yaleB36',...
    'yaleB37','yaleB38', 'yaleB39'};

imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource',...
    'foldernames');

% split into two datastores
% split imds into training and test sets
[Train Test] = splitEachLabel(imds,0.7,'randomized');

Train = shuffle(Train);
Test = shuffle(Test);
Y = Train.Labels;
Y1 = Test.Labels;
L = length(Y);
L1 = length(Y1);


%% COIL100 Object database
% https://www.cs.columbia.edu/CAVE/software/softlib/coil-100.php
% create the folder COIL100 and store the dataset there

rootFolder = 'COIL100';

categories = {'obj1','obj2','obj3','obj4','obj5','obj6','obj7','obj8',...
    'obj9','obj10', 'obj11','obj12','obj13','obj14','obj15','obj16','obj17','obj18',...
    'obj19','obj20', 'obj21','obj22','obj23','obj24','obj25','obj26','obj27','obj28',...
    'obj29','obj30', 'obj31','obj32','obj33','obj34','obj35','obj36','obj37','obj38',...
    'obj39','obj40','obj41','obj42','obj43','obj44','obj45','obj46','obj47','obj48',...
    'obj49','obj50', 'obj51','obj52','obj53','obj54','obj55','obj56','obj57','obj58',...
    'obj59','obj60','obj61','obj62','obj63','obj64','obj65','obj66','obj67','obj68',...
    'obj69','obj70','obj71','obj72','obj73','obj74','obj75','obj76','obj77','obj78',...
    'obj79','obj80','obj81','obj82','obj83','obj84','obj85','obj86','obj87','obj88',...
    'obj89','obj90','obj91','obj92','obj93','obj94','obj95','obj96','obj97','obj98',...
    'obj99','obj100'};
imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource',...
    'foldernames');




% split imds into training and test sets
[Train Test] = splitEachLabel(imds,0.7,'randomized');
countEachLabel(Train);
countEachLabel(Test);
Train = shuffle(Train);
Test = shuffle(Test);

% Extract Labels for Training and Test sets
Y = Train.Labels;
Y1 = Test.Labels;
L = length(Y);
L1 = length(Y1);
% 
rng default; % to facilitate the repetition of results

%% Following is the processing of the dataset chosen
nc = 256;  % Number of columns of image. Images will be resized to the designed pixel array size 256x256
nr = 256;  % Number of rows of image
Xs = zeros(L,nc);   % Matrix to hold compressed samples of Training set
XsT = zeros(L1,nc); % Matrix to hold compressed samples of Test set

% Generate two pseudo-random sequences using ECA rule30
seq = nc*2;  % seed length 512.
seed = randi([0 1],1,seq);  % seed generated
ECA = elementaryCellularAutomata(30, seq, seed); % gives matrix of 512*512 to be split into two equal halves
ECA1 = ECA(1:256,1:256);     % utilized for random selection of pixels (alpha bits)
ECA2 = ECA(257:512,257:512); % utilized for random modulation of pixels (beta bits)


% FPN due to the stacking of 5-ADCs-- For normal operation, this part can be commented
*******************************************************************************************************************************
Delta = 0.17; % Scale this to image level intensity
Delta1 = Delta*255/2.5;
pattern_length = 5;    % pattern repeats every 5 columns
% Initialize addition pattern with zeros (no change by default)
pattern = zeros(1, nc);
% Fill pattern: skip 1st, 6th, 11th, ...; add scaled values to next 4 columns;
% 1st column is nominal, 2nd column Delta (Delta1--scaled) is added, 3rd column 2*Delta added, 4th column 3*Delta added and in 5th column 4*Delta added 
for i = 1:pattern_length:nc
    cols = (i+1):(i+pattern_length-1);   % next 4 columns
    cols = cols(cols <= nc);       % avoid exceeding last column
    pattern(cols) = (1:length(cols)) * Delta1; % scaled additions
end
******************************************************************************************************************************* 


for i=1:L
    img = readimage(Train,i); % read the image from the training set
    if ndims(img)==3
        img = rgb2gray(img); % Convert the image to gray if it is color (GIT and COIL100 datasets)
    end
    img1 = imresize(img,[nr nc]); % Resize the image to the size of pixel array
    img1 = double(img1); % to perform multiplication with other matrix
    % to include five-column fpn
    % img1 = img1+pattern;  % pattern extends to all rows

    % random pixel selection used with sigma_delta_RTMM and sigma_delta_RBMM. 
    % The following line is commented when sigma_delta_RBBMM is used.
    img2 = img1.*ECA1;

    % If sigma_delta_RBBMM is used, then uncomment the following line
   %img2= img1;

   
    m = sigma_delta_RTMM(img2,127,ECA1,ECA2);   % For using random ternary measurement matrix
%    m = sigma_delta_RBBMM(img2,127,ECA2);      % For using random bipolar binary measurement matrix
%    m = sigma_delta_RBMM(img2,127,ECA1);       % For using random binary measurement matrix
     
    Xs(i,:) = m; % Average value of each column is a CS sample and is stored in  the.
                % store the feature vector for each training set image

end
    
for i=1:L1
    img = readimage(Test,i); % read the image from the test set
    if ndims(img)==3
        img = rgb2gray(img); % Convert the image to gray if it is color
    end
    img1 = imresize(img,[nr nc]);
    img1 = double(img1);
  
    % to include five-column fpn
    % img1 = img1+pattern;  % pattern extends to all rows

% random pixel selection used with sigma_delta_RTMM and sigma_delta_RBMM. 
% The following line is commented when sigma_delta_RBBMM is used.
    img2 = img1.*ECA1;

% If sigma_delta_RBBMM is used, then uncomment the following line
   %img2= img1;

    m = sigma_delta_RTMM(img2,127,ECA1,ECA2);   % For using random ternary measurement matrix
%    m = sigma_delta_RBBMM(img2,127,ECA2);      % For using random bipolar binary measurement matrix
%    m = sigma_delta_RBMM(img2,127,ECA1);       % For using random binary measurement matrix
     
    XsT(i,:) = m;   % store the feature vector for each test set image
end

% to plot the histograms of CS sample matrices to observe the distribution
% and the number of bits required to represent extreme-size CS samples
tiledlayout('flow')   
nexttile
histogram(Xs)
nexttile
histogram(XsT)

% define the template for SVM and standardize the results
t = templateSVM('KernelFunction','Linear','Standardize',true);%,'BoxConstraint',141.98,'KernelScale',0.1715);
mdl = fitcecoc(Xs,Y,'Coding','onevsall','Learners',t); % Fit the data i.e. perform the training
pred = predict(mdl,XsT);  % Prediction on the test dataset
acc = sum(pred==Y1)/L1   % Accuracy for the prection


