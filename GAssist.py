# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# Name: GAssist.py

# Import modules
from GAssist_Pop import *
from GAssist_Windowing import *  # might not be needed here !!!!!!!check
from GAssist_Constants import *
import time

class GAssist:
    def __init__(self, e, outFileString, popOutFileString, bitLength, CVparts, graphPerform):
        """ Initialize GAssist algorithm objects:  """
        # Output Filenames
        self.outFile = outFileString  # Saves tracking values
        self.popFile = popOutFileString  # Saves evaluation and print out of final populations (at pre-specified iteration stops)

        # Board Parameters
        self.env = e

        # Used do determine how assessments done.  is there testing data? if CVparts >1 than yes.
        self.CVpartitions = CVparts

        # Track learning time
        self.startLearningTime = 0
        self.endLearningTime = 0
        self.evalTime = 0

        # Other Objects
        self.attributeGenList = []
        self.performanceTrackList = [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]
        self.graphPerform = graphPerform
        self.bitLength = bitLength
        self.trainAccuracy = 0
        self.testAccuracy = 0

        # Initialization Parameters
        self.smartInit = False
        self.cwInit = False

    # METHODS SETTING PARAMETERS *****************************************************************************
    def setNumberOfTrials(self, trials, iterList):
        """ Resets the maximal number of learning iterations in an LCS run."""
        self.maxProblems = trials
        self.iterStops = iterList

    def setTrackingIterations(self, trkCyc):
        """ Resets the parameters for how often progress tracking is reset and sent to the output file."""
        self.trackIter = trkCyc

    def setInitialization(self, initMethod):
        """ Sets the rule initialization method.  Can be 'none' which is a random initialization, 'smart' which is similar to standard covering in XCS
        and 'cw' which is similar to smart, but with equal class opportunity. """

        if initMethod == "smart":
            self.smartInit = True
            self.cwInit = False
            print("Smart initialization used!")
        elif initMethod == "cw":
            self.smartInit = True
            self.cwInit = True
            print("Smart and balanced initialization used!")
        else:
            self.smartInit = False
            self.cwInit = False
            print("Default random initialization used!")

    # METODO CLAVE*********************************************************************************************
    def runGAssist(self):
        """ Method to run the GAssist algorithm. """
        # ***************************************************************
        try:
            pW = open(self.outFile + '.txt', 'w')
        except IOError as e:
            print(f"I/O error({e.errno}): {e.strerror}")
            raise
        # ***************************************************************

        # Initialize Windowing - takes all data set instances as argument
        windows = Windowing(self.env, self.maxProblems)

        # Initialize the population
        population = GAssistPop(self.env, self.env.getTrainSet(), self.env.getTestSet(), windows, self.smartInit, self.cwInit, self.maxProblems)

        # Get and print initial tracking stats
        #pW.write("Iteration\tWindow\tBestFitness\tBestAccuracy\tBestNumRules\tBestAliverules\tBestGenerality\tAverageFitness\tAverageAccuracy\tAverageNumRules\tAverageNumRulesUtils\tAverageGenerality\n") # Uncomment if you want to see the results in English (Descomentar si deseas ver los resultados en ingles)
        pW.write("Iteración\tVentana\tMejorAptitud\tMejorPrecisión\tMejorNúmeroDeReglas\tMejorReglasActivas\tMejorGeneralidad\tPromedioAptitud\tPromedioPrecisión\tPromedioNúmeroDeReglas\tPromedioNúmeroDeReglasUtilizadas\tPromedioGeneralidad\n") # Uncomment if you want to see the results in Spanish (Descomentar si deseas ver los resultados en español)
        trackInit = population.getTracking()  # an initial evaluation should be done using the entire set. also - send window the iteration, instead of separate internal counter
        currentWindow = windows.getCurrentVersion()
        self.writePerformance(pW, trackInit, -1, currentWindow)  # Again this may need to be removed here - initial population performance writing.

        # TIMER STARTS
        self.startLearningTime = time.time()
        tempTimeA = 0
        tempTimeB = 0

        iteration = 0
        print("Begin Evolution Cycles:")
        while iteration < self.maxProblems:
            # Run a single evolution iteration
            lastIteration = (iteration == self.maxProblems - 1)

            cons.setLearning(iteration / self.maxProblems)
            windows.setEvalScope(False)  # evaluations will take place within a window.
            population.evolvePopulation(lastIteration, iteration)
            # Track the population statistics
            track = population.getTracking()
            if (iteration + 1) % self.trackIter == 0:
                self.writePerformance(pW, track, iteration, windows.getCurrentVersion())

            # Evaluate current status at pre-specified stop iterations.
            if iteration + 1 in self.iterStops:
                tempTimeA = time.time()
                # ***************************************************************
                try:
                    popW = open(self.popFile + '.' + str(iteration + 1) + '.txt', 'w')
                except IOError as e:
                    print(f"I/O error({e.errno}): {e.strerror}")
                    raise
                # ***************************************************************

                windows.setEvalScope(True)  # evaluations will take place within the entire training set

                population.trainPopulation()
                track = population.getTracking()

                bestAgent = population.getBest()
                bA = bestAgent.clone()

                trainEval = track
                testEval = []
                if self.CVpartitions > 1:
                    # Evaluate best agent using the test set.
                    inst = self.env.getTestSet()
                    testAccuracy = bA.testEvaluateAgent(inst, population.globalMDL)
                    testEval = [testAccuracy]
                else:
                    testEval = [None]

                # Time management
                self.endLearningTime = time.time()
                tempTimeB = time.time()
                self.evalTime += tempTimeB - tempTimeA
                learnSec = self.endLearningTime - self.startLearningTime - self.evalTime

                self.printCSEvaluation(popW, trainEval, testEval, learnSec, self.env, self.CVpartitions, bestAgent, iteration)

                # ***************************************************************
                try:
                    popW.close()
                except IOError as e:
                    print(f"I/O error({e.errno}): {e.strerror}")
                    raise
                # ***************************************************************
                population.returnPopulation()
            iteration += 1

        # ***************************************************************
        print("LCS Training and Evaluation Complete!")
        try:
            pW.close()
        except IOError as e:
            print(f"I/O error({e.errno}): {e.strerror}")
            raise
        # ***************************************************************

    def printCSEvaluation(self, popW, track, testEval, learnSec, env, CVPartition, bestAgent, iteration):
        """ Makes an output file containing a complete evaluation of the current GALE agent population, and a print out of the current best agent (and all its rules). """
        vector = bestAgent.getVector()
        print("Stop Point Agent Evaluation::")
        # Two sections to the output: Learning Characteristics and Population/Agent Characteristics
        popW.write("Training:\n")
        #popW.write("Iteration\tBestFitness\tBestAccuracy\tBestNumRules\tBestAliverules\tBestGenerality\tAverageFitness\tAverageAccuracy\tAverageNumRules\tAverageNumRulesUtils\tAverageGenerality\tRunTime(min)\n") # Uncomment if you want to see the results in English (Descomentar si deseas ver los resultados en ingles)
        popW.write("Iteración\tMejorAptitud\tMejorPrecisión\tMejorNúmeroDeReglas\tMejorReglasVivas\tMejorGeneralidad\tAptitudPromedio\tPrecisiónPromedio\tNúmeroDeReglasPromedio\tNúmeroDeReglasVivasPromedio\tGeneralidadPromedio\tTiempoDeEjecución(min)\n") # Uncomment if you want to see the results in Spanish (Descomentar si deseas ver los resultados en español)
        popW.write(str(iteration + 1) + "\t" + str(track[0]) + "\t" + str(track[1]) + "\t" + str(track[2]) + "\t" + str(track[3]) + "\t" + str(track[4]) + "\t" + str(track[5]) + "\t" + str(track[6]) + "\t" + str(track[7]) + "\t" + str(track[8]) + "\t" + str(track[9]) + "\t" + str(learnSec / 60) + "\n")
        #print(f"Eval. Point: {iteration + 1}\t Window: All\t BestFitness: {track[0]}\t BestAccuracy: {track[1]}\t BestNumRules: {track[2]}\t BestAliveRules: {track[3]}\t BestGenerality: {track[4]}\t AveFitness: {track[5]}\t AveAccuracy: {track[6]}\t AveNumRules: {track[7]}\t AveNumRulesUtils: {track[8]}\t AveGenerality: {track[9]}") # Uncomment if you want to see the results in English (Descomentar si deseas ver los resultados en ingles)
        print(f"Punto de Evaluación: {iteration + 1}\t Ventana: Todas\t MejorAptitud: {track[0]}\t MejorPrecisión: {track[1]}\t MejorNúmeroDeReglas: {track[2]}\t MejorReglasVivas: {track[3]}\t MejorGeneralidad: {track[4]}\t AptitudPromedio: {track[5]}\t PrecisiónPromedio: {track[6]}\t NúmeroDeReglasPromedio: {track[7]}\t NúmeroDeReglasVivasPromedio: {track[8]}\t GeneralidadPromedio: {track[9]}") # Uncomment if you want to see the results in Spanish (Descomentar si deseas ver los resultados en español)
        #popW.write("Testing Accuracy:\n") # Uncomment if you want to see the results in English (Descomentar si deseas ver los resultados en ingles)
        popW.write("Precisión de Testing:\n") # Uncomment if you want to see the results in Spanish (Descomentar si deseas ver los resultados en español)
        if CVPartition > 1:  # Print testing stats if CV has been utilized
            inst = self.env.getTestSet()  # Obtener el conjunto de test
            y_true = [int(instance[1]) for instance in inst]  # Convertir a enteros
            y_pred = [int(bestAgent.classify(instance)) for instance in inst]  # Convertir a enteros
            
            # Calcular la matriz de confusión
            matriz_confusion = confusion_matrix(y_true, y_pred)
            print("Matriz de Confusión (Test Set):")
            print(matriz_confusion)
            
            # Imprimir la matriz de confusión en el archivo
            popW.write("Matriz de Confusión (Test Set):\n")
            popW.write(str(matriz_confusion) + "\n")
            
            # Calcular métricas adicionales: Precisión, Recall y F1 Score
            precision = precision_score(y_true, y_pred, pos_label=1)
            recall = recall_score(y_true, y_pred, pos_label=1)
            f1 = f1_score(y_true, y_pred, pos_label=1)
            
            # Imprimir las métricas adicionales
            popW.write("Precisión de Testing:\n")
            popW.write(str(precision) + "\n")
            popW.write("Recall de Testing:\n")
            popW.write(str(recall) + "\n")
            popW.write("F1 Score de Testing:\n")
            popW.write(str(f1) + "\n")
            
            print(f"Precisión de Testing: {precision}")
            print(f"Recall de Testing: {recall}")
            print(f"F1 Score de Testing: {f1}")
        else:
            popW.write("NA\n")
        popW.write("Population Characterization:\n")
        popW.write("WildSum\n")

        # Print the attribute labels
        headList = env.getHeaderList()
        for i in range(len(headList) - 1):  # Added the -1 to get rid of the Class Header
            if i < len(headList) - 2:
                popW.write(str(headList[i]) + "\t")
            else:
                popW.write(str(headList[i]) + "\n")

        wildCount = self.characterizePop(vector)
        self.attributeGenList = self.condenseToAttributes(wildCount, self.bitLength)  # List of the bitLength corrected wild counts for each attribute.

        # Prints out the generality count for each attribute.
        for i in range(len(self.attributeGenList)):
            if i < len(self.attributeGenList) - 1:
                popW.write(str(self.attributeGenList[i]) + "\t")
            else:
                popW.write(str(self.attributeGenList[i]) + "\n")

        popW.write("Ruleset Population: \n")
        popW.write("Condition\tAction\tCorrect\tMatch\tAccuracy\tGenerality\n")

        # Prints out the rules of the best agent.
        for i in range(len(vector)):
            genCount = 0
            for j in range(len(vector[i])):
                if j == len(vector[i]) - 1:  # Write Action/Class
                    popW.write("\t")
                    popW.write(str(vector[i][j]))
                else:  # Write Condition
                    popW.write(str(vector[i][j]))
                    if vector[i][j] == cons.dontCare:
                        genCount += 1
            tempAcc = 0
            if bestAgent.agnPer.utilRules[i] == 0:
                tempAcc = 0
            else:
                tempAcc = bestAgent.agnPer.accurateRule[i] / float(bestAgent.agnPer.utilRules[i])
            popW.write("\t" + str(bestAgent.agnPer.accurateRule[i]) + "\t" + str(bestAgent.agnPer.utilRules[i]) + "\t" + str(tempAcc) + "\t" + str(genCount / float(len(vector[i]) - 1)) + "\n")

        if cons.defaultClass != "disabled":
            # Print the default rule.
            for i in range(len(vector[0]) - 1):
                popW.write(cons.dontCare)
            popW.write("\t" + str(bestAgent.defaultClass))

            tempAcc = 0
            i = len(vector)
            if bestAgent.agnPer.utilRules[i] == 0:
                tempAcc = 0
            else:
                tempAcc = bestAgent.agnPer.accurateRule[i] / float(bestAgent.agnPer.utilRules[i])
            popW.write("\t" + str(bestAgent.agnPer.accurateRule[i]) + "\t" + str(bestAgent.agnPer.utilRules[i]) + "\t" + str(tempAcc) + "\t" + str(1) + "\n")

    def characterizePop(self, vector):
        """ Make a list counting the #'s in each attribute position across the best agent's rule set. """
        countList = []
        for x in range(len(vector[0]) - 1):
            countList.append(int(0))
        for i in range(len(vector)):
            for j in range(len(vector[i]) - 1):
                if vector[i][j] == cons.dontCare:
                    countList[j] += 1
        return countList

    def condenseToAttributes(self, tempList, bitLength):
        """ Takes the results of 'characterizePop' and condenses the count values down to include all bits coding a single attribute """
        temp = 0
        newList = []
        for i in range(len(tempList)):
            if (i + 1) % int(bitLength) != 0:  # first run it will be 1
                temp += tempList[i]
            else:
                newList.append(temp + tempList[i])
                temp = 0
        return newList

   
 
    def writePerformance(self, pW, track, iteration, currentWindow):
        """ Writes the tracking data.  """
        pW.write(str(iteration + 1) + "\t" + str(currentWindow) + "\t" + str(track[0]) + "\t" + str(track[1]) + "\t" + str(track[2]) + "\t" + str(track[3]) + "\t" + str(track[4]) + "\t" + str(track[5]) + "\t" + str(track[6]) + "\t" + str(track[7]) + "\t" + str(track[8]) + "\t" + str(track[9]) + "\n")
        print("Iteración: " + str(iteration + 1) + "\t Ventana: " + str(currentWindow) + "\t MejorAptitud: " + str(track[0]) + "\t MejorPrecisión: " + str(track[1]) + "\t MejorNumReglas: " + str(track[2]) + "\t MejorReglasVivas: " + str(track[3]) + "\t MejorGeneralidad: " + str(track[4]) + "\t PromedioAptitud: " + str(track[5]) + "\t PromedioPrecisión: " + str(track[6]) + "\tPromedioNumReglas: " + str(track[7]) + "\t PromedioNumReglasUtilizadas: " + str(track[8]) + "\t PromedioGeneralidad: " + str(track[9]))
        #"Iteration\tBestFitness\tBestAccuracy\tBestNumRules\tBestAliverules\tBestGenerality\tAverageFitness\tAverageAccuracy\tAverageNumRules\tAverageNumRulesUtils\tAverageGenerality\n"

    # DEBUGGING
    def printCurrentBest(self, bestAgent):
        """ Used for debugging - prints out the best agent of the current iteration. """
        vector = bestAgent.getVector()

        print("Ruleset Population:")
        print("Condition\tAction\tCorrect\tMatch\tAccuracy")

        # Prints out the rules of the best agent.
        for i in range(len(vector)):
            print(len(vector))
            print(len(bestAgent.agnPer.utilRules))
            tempAcc = 0
            if bestAgent.agnPer.utilRules[i] == 0:
                tempAcc = 0
            else:
                tempAcc = bestAgent.agnPer.accurateRule[i] / float(bestAgent.agnPer.utilRules[i])

            print(str(vector[i]) + "\t" + str(bestAgent.agnPer.accurateRule[i]) + "\t" + str(bestAgent.agnPer.utilRules[i]) + "\t" + str(tempAcc))

        # Print the default rule.
        if cons.defaultClass != "disabled":
            tempAcc = 0
            i = len(vector)
            if bestAgent.agnPer.utilRules[i] == 0:
                tempAcc = 0
            else:
                tempAcc = bestAgent.agnPer.accurateRule[i] / float(bestAgent.agnPer.utilRules[i])

            print("Default Rule #### - " + str(bestAgent.defaultClass) + "\t" + str(bestAgent.agnPer.accurateRule[i]) + "\t" + str(bestAgent.agnPer.utilRules[i]) + "\t" + str(tempAcc) + "\t" + str(1))
