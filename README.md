# bootstrap-rom3d

This repository contains python programs developed for training and testing of ROMs and are a part of a preprint submitted to the journal Computers and Geosciences titled "A Bootstrap Strategy to Train, Validate and Test Reduced Order Models of Coupled Geomechanical Processes".

Dependencies: Python 3.6 and later, Tensorflow (tested on 2.0), numpy, matplotlib, os, sklearn, termcolor, pickle, pyvista

Important: Data for all the programs are already included in designated folders. The programs should automatically load them while running. For 3D visualisations VisIt is preferred.


All the operations (including creation of results folder, plotting and saving) have been automated. Please run the programs in this sequence.
1) "trainROM.py" : This is used to train the Reduced Order Model using the already extracted weights from the full order simulation states.
2) "predictUsingROM.py" : This must be used to load the desired BCs and use the trained ROM from step 1 to generate forecasts.
3) "plotResults.py" : Shows the progress in training w.r.t. the number of simulations.


![image1](https://user-images.githubusercontent.com/113099597/189470155-04a85fc7-05e5-4f08-8676-dd72755b5a35.png)
