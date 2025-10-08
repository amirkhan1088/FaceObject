% Download store the databases in the local memory in the same folder and
% change the path of rootFolder accordingly the final folder name is
% already mentioned
% Run for one database at a time
%% GIT face database LInk
%https://www.anefian.com/research/face_reco.htm
categories = {'s01','s02','s03','s04','s05','s06','s07','s08','s09','s10',...
    's11','s12','s13','s14','s15','s16','s17','s18','s19','s20',...
    's21','s22','s23','s24','s25','s26','s27','s28','s29','s30',...
    's31','s32','s33','s34','s35','s36','s37','s38','s39','s40',...
    's41','s42','s43','s44','s45','s46','s47','s48','s49','s50'};

rootFolder = 'gt_db';
imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource',...
    'foldernames');  % create the imagestore from the dataset

% split imds into trian and test
[Train Test] = splitEachLabel(imds,0.55,'randomized');
% this ratio of 0.55 gives 8 images in train set and 7 images in test set
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
% http://vision.ucsd.edu/~leekc/ExtYaleDatabase/ExtYaleB.html % accessed in
% 2021
% create the folder ExtYaleB and store the dataset there
rootFolder = 'ExtYaleB'; % without ambient image files 

categories = {'yaleB11','yaleB12', 'yaleB13', 'yaleB15','yaleB16',...
    'yaleB17','yaleB18', 'yaleB19', 'yaleB20','yaleB21', 'yaleB22', 'yaleB23',...
    'yaleB24','yaleB25','yaleB26','yaleB27', 'yaleB28', 'yaleB29','yaleB30',...
    'yaleB31','yaleB32', 'yaleB33', 'yaleB34', 'yaleB35','yaleB36',...
    'yaleB37','yaleB38', 'yaleB39'};

imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource',...
    'foldernames');
% number of samples
L1 = length(imds.Labels);
% split in two datastores
% split imds into two trian and test
[Train Test] = splitEachLabel(imds,0.7,'randomized');
%
Train = shuffle(Train);
Test = shuffle(Test);
Y = Train.Labels;
Y1 = Test.Labels;
L = length(Y);
L1 = length(Y1);
%% COIL100 Object database
%https://www.cs.columbia.edu/CAVE/software/softlib/coil-100.php
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

%Load test data
% rootFolder = 'cifar10Test';
% imds_test = imageDatastore(fullfile(rootFolder, categories), ...
%     'LabelSource', 'foldernames');
% Split Dataset into train and test
[Train Test] = splitEachLabel(imds,0.7,'randomized');
countEachLabel(Train);
countEachLabel(Test);
Train = shuffle(Train);
Test = shuffle(Test);

% Sampling gray images
Y = Train.Labels;
Y1 = Test.Labels;
L = length(Y);
L1 = length(Y1);
%% Following is the processing of dataset chosen
nc = 256;  % Number of columns of image
nr = 256;  % Number of rows of image
Xs = zeros(L,nc);  % Matrix to hold compressed samples of Training set
XsT = zeros(L1,nc); %Matrix to hold compressed samples of Test set

c = 0.5; %Percentage of pixels set to zero

Row = zeros(nr,nc); %  matrix having indices of image randomly distributed across each row
for i=1:nr
     Row(i,:) = randperm(nc);
end

mu1 = randsrc(nr,nc,[-1 1]); % random bipolar binary matrix
c1 = ceil(c*nr); % 50% rows elements to be set to zero
 
for i=1:L
    img = readimage(Train,i); % read the image from the training set
    if ndims(img)==3
        img = rgb2gray(img); % Convert the image to gray if it is color (GIT and COIL100 datasets)
    end
    img1 = imresize(img,[nr nc]); % Resize the image to the size of pixel array
    img1 = double(img1); % to perform multiplication with other matrix
    img1 = img1.*mu1;    % each pixel of the image is randomly multiplied by 1 or -1

    Zero_ind = zeros(c1,nc); % matrix storing the zero indices to be fed to sigma-delta ADC to skip that pixel and does not perform any operation 
    for k=1:nc
        sel=Row(1:c1,k);  % vector containg the indices of each column to be made to zero
        Zero_ind(:,k) = sel; % Also save these indices to Zero_ind holder
        img1(sel,k) = 0;    % Selecing the pixel zero corresponding to vector sel

    end
    m = sigma_delta_UD_Counter_col_not_selected_skipped(img1,127,Zero_ind);% this function is explained in the separate file
    Xs(i,:) = m; % Average value of each column is a CS samples and is stored in  the 

end
    
for i=1:L1
    img = readimage(Test,i); % read the image from the test set
    if ndims(img)==3
        img = rgb2gray(img); % Convert the image to gray if it is color
    end
    img1 = imresize(img,[nr nc]);
    img1 = double(img1);
    img1 = img1.*mu1;
    
    for k=1:nc
        sel=Row(1:c1,k);
        Zero_ind(:,k) = sel;
        img1(sel,k) = 0;
        
    end
    m = sigma_delta_UD_Counter_col_not_selected_skipped(img1,127,Zero_ind);
    XsT(i,:) = m;
end

% to plot the histograms of CS sample matrices to observe the distribution
% and the number of bits required to represent extreme size CS samples
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

