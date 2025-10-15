# Face/Object Recognition using Compressed Sensing
This repository contains files to perform face and object recognition using compressed sensing.
Primary files are main.m and custom sigma-delta ADC functions to implement three sensing matrices, namely Random Ternary MM, Random Binary MM and Random Bipolar Binary MM, respectively: sigma_delta_RTMM.m,sigma_delta_RBMM.m and sigma_delta_RBBMM.m

The code in main.m utilizes an elementary cellular automata-based pseudo random number generator whose code can be downloaded from the link: https://uk.mathworks.com/matlabcentral/fileexchange/26929-elementary-cellular-automata
The main.m file also includes the code snippet to find the 5-column FPN, possibly the outcome of stacking 5-ADCs of the proposed architecture. The original ADC was proposed in the work, "Khan, A., Fernández-Berni, J., & Carmona-Galán, R. (2023, December). Hardware-Efficient Random-Modulation ΣΔ ADC for Per-Column CS Generation in Vision Sensor. In 2023 30th IEEE International Conference on Electronics, Circuits and Systems (ICECS) (pp. 1-5)."
This code snippet is commented out for normal simulations.

Links for three databases are as follows:
(i) Georgia Tech Face Dataset: https://www.anefian.com/research/face_reco.htm
(ii) Extended Yale B: http://vision.ucsd.edu/datasets/extended-yale-face-database-b-b
(iii) COIL100 Object Dataset: https://www.cs.columbia.edu/CAVE/software/softlib/coil-100.php

Download the database and place it in the local memory, and use the same root folder as specified in the main.m program file.
Run one database loading at a time, and then run the program that performs the training and testing. No random seed has been set so that different runs will yield different accuracies, and the average over 10 (or any other suitable value) runs can be evaluated. In each run, it is also better to rerun the database code block to get a different distribution of training and test databases.

