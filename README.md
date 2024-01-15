# Face_object
This repository contains files to perform face and object recognition using compressed sensing.
It has two files main.m and custom sigma-delta ADC function sigma_delta_UD_Counter_col_not_selected_skipped.m
In the main.m file all the links for three databases are given. Download the the database and keep it in local memory and use the same root folder as used in the program.
Run one database loading at a time and then run the program which performs the training and testing. No random seed has been set so that the different runs will give different accuracy and average over certain say 10 runs can be evaluated. In each run, it is also better to run the database code block to get the new database.

