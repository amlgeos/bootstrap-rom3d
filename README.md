# bootstrap-rom3d

This repository contains python programs developed for training and testing of ROMs and are a part of a preprint submitted to the journal Computers and Geosciences, titled "A Bootstrap Strategy to Train, Validate and Test Reduced Order Models of Coupled Geomechanical Processes".

Dependencies: Python 3.6 and later, Tensorflow (tested on 2.0), numpy, matplotlib, os, sklearn, termcolor, pickle, pyvista

Important: Data for all the programs are already included in designated folders. The programs should automatically load them while running. For 3D visualisations VisIt is preferred.

Prerequisites: 
1) There are part files in the folder "./analysis/u/" with a prefix "part". Please use the following command in the Linux terminal to join them: cat part* > pca_.npy. Let the joined file remain there itself.
2) There are part files in the folder "./results/y_val/" with a prefix "yv". Please use the following command in the Linux terminal to join them: cat yv* > y_val_seq.pkl
3) There are part files in the folder "./results/y_val_pred/" with a prefix "yvp". Please use the following command in the Linux terminal to join them: cat yvp* > y_val_pred_seq.pkl
4) There are part files in the folder "./results/y_test/" with a prefix "yt". Please use the following command in the Linux terminal to join them: cat yt* > y_test_seq.pkl
5) There are part files in the folder "./results/y_test_pred/" with a prefix "ytp". Please use the following command in the Linux terminal to join them: cat ytp* > y_test_pred_seq.pkl

After joining files in steps 2 to 5 put each of them inside "./analysis/results/". 




All the operations (including creation of results folder, plotting and saving) have been automated. Please run the programs in this sequence.
1) "trainROM.py" : This is used to train the Reduced Order Model using the already extracted weights from the full order simulation states.
2) "predictUsingROM.py" : This must be used to load the desired BCs and use the trained ROM from step 1 to generate forecasts.
3) "plotResults.py" : Shows the progress in training w.r.t. the number of simulations.


![image1](https://user-images.githubusercontent.com/113099597/189470334-b1d89245-d0e9-4c47-a750-21ac3da92237.png)
