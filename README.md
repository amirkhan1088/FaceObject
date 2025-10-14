# Face_object
This repository contains files to perform face and object recognition using compressed sensing. Primary files are main.m and custom sigma-delta ADC functions to implement three sensing matrices namely Random Ternary MM, Random Binary MM and Random Bipolar Binary MM, respectively: sigma_delta_RTMM.m,sigma_delta_RBMM.m and sigma_delta_RBBMM.m

The code in main.m utilizes an elementary cellular automata-based pseudo random number generator whose code can be downloaded from the link: https://uk.mathworks.com/matlabcentral/fileexchange/26929-elementary-cellular-automata

Links for three databases are given below: 
(i) Georgia Tech Face Dataset: https://www.anefian.com/research/face_reco.htm 
(ii) Extended Yale B: http://vision.ucsd.edu/datasets/extended-yale-face-database-b-b 
(iii) COIL100 Object Dataset: https://www.cs.columbia.edu/CAVE/software/softlib/coil-100.php

Download the database and place it in the local memory, and use the same root folder as used in the main.m program file. Run one database loading at a time and then run the program which performs the training and testing. No random seed has been set so that the different runs will give different accuracy and average over certain say 10 runs can be evaluated. In each run, it is also better to run the database code block to get the different distribution of training and test databases.

