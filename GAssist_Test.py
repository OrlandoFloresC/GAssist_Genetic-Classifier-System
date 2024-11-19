# -*- coding: utf-8 -*-
#!/usr/bin/env python3
#Name: GAssist_Test.py

# Import modules
from GAssist import *
from GAssist_Environment import *
from GAssist_Constants import *
import pandas as pd
from sklearn.model_selection import train_test_split

# Recordar borrar los print en "GAssist_Environment.py"

def main():
    """ Runs independently from command line to test the GAssist algorithm. """
    graphPerformance = False  # Built-in graphing ability, currently not functional, but mechanism is in place.

    trainData = "2_1000_0_1600_0_0_CV_0_Train.txt"  # Comentar este si vas a hacer pruebas con tu propio dataset
    testData = "2_1000_0_1600_0_0_CV_0_Test.txt" # Comentar este si vas a hacer pruebas con tu propio dataset
    #trainData = "train_data_aligned.txt" # Descomentar este si vas a hacer pruebas con tu propio dataset
    #testData = "test_data_aligned.txt"   # Descomentar este si vas a hacer pruebas con tu propio dataset
    outProg = "GH_GAssist_ProgressTrack"
    outPop = "GH_GAssist_PopulationOut"
    bitLength = 1  # This implementation is not yet set up to handle other rule representations, or bit encoding lengths.
    CVpartitions = 10
    trackCycles = 1

    # Run Parameters - User specified.
    iterInput = '20.50.100'
    pop = 100
    wild = 0.5
    defaultClass = "0"  # auto, 0, disabled   
    init = "cw"  # 'none', 'smart', 'cw'
    MDL = 1
    windows = 2

    # Figure out the iteration stops for evaluation, and the max iterations.
    iterList = list(map(int, iterInput.split('.')))
    lastIter = iterList[-1]

    # Sets up algorithm to be run.
    e = GAssist_Environment(trainData, testData, bitLength, init)
    cons.setConstants(pop, wild, defaultClass, e, MDL, windows)
    sampleSize = e.getNrSamples()
    gassist = GAssist(e, outProg, outPop, bitLength, CVpartitions, graphPerformance)

    # Set some GAssist parameters.
    if trackCycles == 'Default':
        gassist.setTrackingIterations(sampleSize)
    else:
        gassist.setTrackingIterations(trackCycles)
    gassist.setNumberOfTrials(lastIter, iterList)
    gassist.setInitialization(init)
    
    # Run the GAssist Algorithm 
    gassist.runGAssist()

if __name__ == '__main__':
    main()

