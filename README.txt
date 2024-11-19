# GAssist Genetic Classifier System 🤖

Welcome to **GAssist Genetic Classifier System**! This repository contains an implementation of GAssist, a rule learning-based classification system that uses genetic algorithms to generate and optimize sets of classification rules.

## Overview 📜
This work is an update to Python 3 of the algorithm originally implemented by **Ryan Urbanowicz in 2010**, which was developed in Python 2. The algorithm was introduced in the article _"Genetic Classifier System as a Heuristic Weighting Method for a Case-Based Classifier System"_. All copyrights belong to the original authors of this work.

## Getting Started 🌐
Below, you can find several options to run GAssist, whether you want to use the provided datasets or your own data, and whether you'd like to adjust the parameters to fit your needs.

#############################################################################

### ⚡ Option 1: Test GAssist with Provided Data

Test the system with the provided datasets: `2_1000_0_1600_0_0_CV_0_est.txt` and `2_1000_0_1600_0_0_CV_0_Train.txt`.

1. Open `GAssist_Test.py`.
2. Make sure that lines 18 and 19 are **uncommented** and lines 20 and 21 are **commented**.
3. Make sure that the train and test files are in the same directory as `GAssist_Test.py`.
4. Run the script from the command line:
   ```sh
   python GAssist_Test.py
   ```
   Or, if you are using a programming environment like Visual Studio Code, run it from "RUN".
5. You should get 3 files named `GH_GAssist_PopulationOut.*` ending with the indices declared in line 29 (`iterInput = '20.50.100'`) and another file named `GH_GAssist_ProgressTrack`. The algorithm will stop at the largest value declared in line 29.
6. Remember that all `.py` files are needed for GAssist to work, and you must have the necessary libraries installed.

### 🚀 Option 2: Test GAssist with Your Own Data

1. Open the file `crear_dataset_discretizado_onehot.py` or `crear_dataset_discretizado_ordinal.py` and add the path of where you have your dataset in line 6.
2. Uncomment lines 43, 35, 38, and 39 to create a discretized set with fewer observations.
   
   **OR**

   Uncomment lines 26, 27, 30, and 31 to create a discretized set with all observations.
3. Run the code.
4. You should get the dataset in `.txt` format as `train_data_aligned` and `test_data_aligned`.
5. Open `GAssist_Test.py`.
6. Make sure that lines 18 and 19 are **commented** and lines 20 and 21 are **uncommented**.
7. Make sure that the train and test files are in the same directory as `GAssist_Test.py`.
8. Run the script from the command line:
   ```sh
   python GAssist_Test.py
   ```
   Or, if you are using a programming environment like Visual Studio Code, run it from "RUN".
9. You should get 3 files named `GH_GAssist_PopulationOut.*` ending with the indices declared in line 29 (`iterInput = '20.50.100'`) and another file named `GH_GAssist_ProgressTrack`. The algorithm will stop at the largest value declared in line 29.
10. Remember that all `.py` files are needed for GAssist to work, and you must have the necessary libraries installed.

### 🛠️ Option 3: Test GAssist with the Whole Dataset

1. Open the file `crear_dataset_discretizado_onehot.py` or `crear_dataset_discretizado_ordinal.py` and add the path of where you have your dataset in line 6.
2. Make sure you have **commented** lines 43, 35, 38, and 39 to create a discretized set with fewer observations.
   
   **OR**

   Make sure you have **commented** lines 26, 27, 30, and 31 to create a discretized set with all observations.
3. Run the code.
4. You should get the dataset in `.txt` format as `train_data_aligned` and `test_data_aligned`.
5. Open `GAssist_Test.py`.
6. Make sure that lines 18 and 19 are **commented** and lines 20 and 21 are **uncommented**.
7. Make sure that the train and test files are in the same directory as `GAssist_Test.py`.
8. Run the script from the command line:
   ```sh
   python GAssist_Test.py
   ```
   Or, if you are using a programming environment like Visual Studio Code, run it from "RUN".
9. You should get 3 files named `GH_GAssist_PopulationOut.*` ending with the indices declared in line 29 (`iterInput = '20.50.100'`) and another file named `GH_GAssist_ProgressTrack`. The algorithm will stop at the largest value declared in line 29.
10. Remember that all `.py` files are needed for GAssist to work, and you must have the necessary libraries installed.

#########################################################################

### 🔨 Option 4: Test GAssist with the Whole Dataset but with Custom Parameters

1. Open the file `crear_dataset_discretizado_onehot.py` or `crear_dataset_discretizado_ordinal.py` and add the path of where you have your dataset in line 6.
2. Make sure you have **commented** lines 43, 35, 38, and 39 to create a discretized set with fewer observations.
   
   **OR**

   Make sure you have **commented** lines 26, 27, 30, and 31 to create a discretized set with all observations.
3. Run the code.
4. You should get the dataset in `.txt` format as `train_data_aligned` and `test_data_aligned`.
5. Open the command line in the same directory where you have all the `.py` files, or if you are using an environment like Visual Studio Code, open the terminal integrated in the directory where you have the GAssist files.
6. From the command line, you should pass the parameters with which you want to run the algorithm. The parameters that you must enter are as follows:

   **Parameters**:
   - **1**: Specify the environment - `'gh'` - genetic heterogeneity SNP datasets with discretely coded genotype attributes and case/control classes.
   - **2**: Training Dataset - `/SomePath/SomeTrainfile.txt`
   - **3**: Testing Dataset - `/SomePath/SomeTestfile.txt`
   - **4**: Output Filename (Learning Progress Tracking) - `/SomePath/SomeProgressName`
   - **5**: Output Filename (Final Pop Evaluation) - `/SomePath/SomeFinalPopName`
   - **6**: Coding Length - `'1'`
   - **7**: Dataset Partitions for CV - `'10'`
   - **8**: Performance Tracking Cycles - `'1'`
   - **9**: Learning iterations - `'100.500.1000'` or for testing `'10.20'`
   - **10**: Initialization - `'none'`, `'smart'`, `'cw'`
   - **11**: PopSize - `'100, 500, 1000'`
   - **12**: Wild frequency - `0.5, 0.75`
   - **13**: Default Class Handling - `auto, 0, disabled`
   - **14**: MDL fitness - `'0'`, `'1'` - where 0 is False, and 1 is True.
   - **15**: Windows - `'1'`, `'2'`, `'4'` - number of partitions of the dataset used in the learning process.

7. Example command to run the main script:
   ```sh
   python GAssist_Main.py gh "C:\path\to\train_data_aligned.txt" "C:\path\to\test_data_aligned.txt" progress_output.txt pop_output.txt 1 10 1 100.500.1000 cw 100 0.5 auto 1 2
   ```
8. You should get 3 files named `pop_output.txt.` ending with the indices declared in `PopSize - '100, 500, 1000'` and another file named `progress_output.txt`. The algorithm stops at the largest value declared in `PopSize - '100, 500, 1000'`, which in this case would be 1000 iterations.
9. Remember that all `.py` files are needed for GAssist to work, and you must have the necessary libraries installed.

## Prerequisites 🛡️
- Python 3.x

## Contributing 💪
Feel free to open issues or pull requests if you'd like to contribute!

## License 🛣️
This work is shared for educational purposes. All rights to the original algorithm belong to the authors of the original work.

Enjoy exploring GAssist! 💻🔍


