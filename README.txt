"This work is an update to Python 3 of the algorithm implemented by Ryan Urbanowicz in 2010, originally developed in Python 2. The algorithm was presented in the article "Genetic Classifier System as a heuristic weighting method for a Case-Based Classifier System". All copyright belongs to the authors of the original work." 

GAssist_Genetic-Classifier-System, is a rule learning-based classification system that uses genetic algorithms to generate and optimize sets of classification rules

If you want to try or use GAssist for your project follow these steps
If you want to use gassis with parameters set by yourself, go to option 4


#############################################################################
Option one; test Gassist.

Test the system with the data provided "2_1000_0_1600_0_0_CV_0_est.txt" and "2_1000_0_1600_0_0_CV_0_Train.txt"

1.- Opens the GAssist_test.py file

2.-  Make sure that the lines of code number 18 and 19 are uncommented and that the 20 and 21 are commented

3.- Make sure that the train and test files are in the same address as the GAssist_Test.py file 

4.- Run from the command line with "python GAssist_Test.py" or if you have some programming environment like Visual Studio Code runs from "RUN"

5.- You should get 3 files with the name "GH_GAssist_PopulationOut." ending with the indices declared in line 29 "iterInput = '20.50.100'" and another file with the name "GH_GAssist_ProgressTrack". The algorithm will stop at the largest value declared in line 29

6.- remember that all the.py files are needed for GAssist to work, and also have libraries necessary for its execution

Option two; test GASIST with own data

1.- Open the file "crear_dataset_discretizado_onehot.py" or "crear_dataset_discretizado_ordinal" and add the address of where you have your data set in line 6.

2.- Descomenta the code line number 43, 35, 38 and 39. This is to create a set discretized with few observations

o
2.- Uncomments the code line number 26, 27, 30 and 31. This is to create a discrete set with few observations

3.- Runs the code

4.- You should get the data set in txt as "train_data_aligned" and "test_data_aligned"

5.- Opens the GAssist_test.py file

6.-  Make sure that the lines of code number 18 and 19 are commented and that the 20 and 21 are uncommented

7.- Make sure that the train and test files are in the same address as the GAssist_Test.py file 

8.- Run from the command line with "python GAssist_Test.py" or if you have some programming environment like Visual Studio Code runs from "RUN"

9.- You should get 3 files with the name "GH_GAssist_PopulationOut." ending with the indices declared in line 29 "iterInput = '20.50.100'" and another file with the name "GH_GAssist_ProgressTrack". The algorithm will stop at the largest value declared in line 29

10.- remember that all the.py files are needed for GAssist to work, and also have libraries necessary for its execution

Option three; test GAssist with the whole dataset

1.- Open the file "crear_dataset_discretizado_onehot.py" or "crear_dataset_discretizado_ordinal" and add the address of where you have your data set in line 6.

2.- Make sure you have commented the lines of code number 43, 35, 38 and 39. This is to create a discrete set with few comments

o
2.- Make sure you have commented lines of code 26, 27, 30 and 31. 

This is to create a discrete set with all the observations

3.- Runs the code

4.- You should get the data set in txt as "train_data_aligned" and "test_data_aligned"

5.- Opens the GAssist_test.py file

6.-  Make sure that the lines of code number 18 and 19 are commented and that the 20 and 21 are uncommented

7.- Make sure that the train and test files are in the same address as the GAssist_Test.py file 

8.- Run from the command line with "python GAssist_Test.py" or if you have some programming environment like Visual Studio Code runs from "RUN"

9.- You should get 3 files with the name "GH_GAssist_PopulationOut." ending with the indices declared in line 29 "iterInput = '20.50.100'" and another file with the name "GH_GAssist_ProgressTrack". The algorithm will stop at the largest value declared in line 29

10.- remember that all the.py files are needed for GAssist to work, and also have libraries necessary for its execution
#########################################################################


Option four; test GAssist with the whole dataset but with my own parameters

1.- Open the file "crear_dataset_discretizado_onehot.py" or "crear_dataset_discretizado_ordinal" and add the address of where you have your data set in line 6.

2.- Make sure you have commented the lines of code number 43, 35, 38 and 39. This is to create a discrete set with few comments

o
2.- Make sure you have commented lines of code 26, 27, 30 and 31. 

This is to create a discrete set with all the observations

3.- Runs the code

4.- You should get the data set in txt as "train_data_aligned" and "test_data_aligned"

5.- Open the command line in the same address where you have all the.py files or if you are using an environment like Visual Visual Sudio Code opens the terminal integrated in the address where you have the GAssist files

6.- From the command line you should psar the parameters with which you want to run the algorithm, the parameters that you must enter are as follows:

Parameters:
    1:  Specify the environment - 'gh' - gh = genetic heterogeneity snp datasets with discretely coded genotype attributes and a case/control classes.
    2:  Training Dataset - '/SomePath/SomeTrainfile.txt'
    3:  Testing Dataset - '/SomePath/SomeTestfile.txt'
    4:  Output Filename (Learning Progress Tracking) - '/SomePath/SomeProgressName'
    5:  Output Filename (Final Pop Evaluation) - '/SomePath/SomeFinalPopName'
    6:  Coding Length - '1'
    7:  Dataset Partitions for CV - '10'
    8:  Performance Tracking Cycles - '1'
    9:  Learning iterations - '100.500.1000' or for testing '10.20'
    10: Initialization - 'none', 'smart', 'cw'
    11: PopSize - '100, 500, 1000'
    12: Wild frequency - 0.5, 0.75
    13: Default Class Handling - auto, 0, disabled  - automatic default class evolution, vs. pre-specify the default class (can also add a method which chooses via which class is most represented.
    14: MDL fitness - '0', '1' - where 0 is False, and 1 is True.
    15: Windows - '1', '2', '4' - number of partitions of the dataset used in the learning process. (higher values tend to promote generalization, and reduce computational time.)
"""

7.- You should enter the command line as the following example:
python GAssist_Main.py gh "C:\path\to\directory\train_data_aligned.txt" "C:\path\to\directory\test_data_aligned.txt" progress_output.txt pop_output.txt 1 10 1 100.500.1000 cw 100 0.5 auto 1 2 

9.- You should get 3 files with the name "pop_output.txt." ending with the indices declared in "PopSize - '100, 500, 1000'" and another file with the name "progress_output.txt". The algorithm stops at the largest value declared in "PopSize - '100, 500, 1000'" in this case would be 1000 iterations

10.- remember that all the.py files are needed for GAssist to work, and also have libraries necessary for its execution

