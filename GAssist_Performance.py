# -*- coding: utf-8 -*-
#!/usr/bin/env python3
#Name: GAssist_Performance.py

# Import modules
from GAssist_Constants import * 
import math
import random

class GAssistPerformance:
    def __init__(self, nClass): 
        self.numAliveRules = 0  # Number of rules that have matched at least one instance.
        self.totalInstances = 0  # Tracks all instances examined
        self.okInstances = 0  # Correct prediction counter
        
        self.accuracy = 0.0
        self.fitness = 0.0

        self.utilRules = []  # Tracks whether the rule was used.
        self.accurateRule = []
        
        self.positionRuleMatch = 0  # New
        self.isEvaluated = False
        self.numClass = nClass  # Number of classes
   
    def resetPerformance(self, numRules):
        """ Reset the performance parameters. """
        self.numAliveRules = 0
        self.totalInstances = 0  # Tracks all instances
        self.okInstances = 0  # Correct prediction counter
        self.accuracy = 0.0
        self.fitness = 0.0  
        self.utilRules = [0] * numRules
        self.accurateRule = [0] * numRules
        self.isEvaluated = False
        
    def addPrediction(self, predicted, real):
        """ Function used to inform PerformanceAgent of each example classified during the training stage. """
        self.totalInstances += 1
        if int(predicted) != -1:
            if self.utilRules[self.positionRuleMatch] == 0 and self.positionRuleMatch < len(self.utilRules):
                self.numAliveRules += 1
            self.utilRules[self.positionRuleMatch] += 1  # Tracks whether the rule was used.
        if int(predicted) == int(real):
            self.okInstances += 1
            self.accurateRule[self.positionRuleMatch] += 1

    def calculatePerformance(self, globalMDL, ind): 
        """ Calculates the accuracy and fitness parameters. """
        self.accuracy = self.okInstances / float(self.totalInstances)
        mdlFit = 0.0
        if cons.useMDL:
            mdlFit = globalMDL.mdlFitness(ind)
        
        penalty = 1.0
        if self.numAliveRules < cons.sizePenaltyMinRules:  # Agents have fitness penalty if the number of rules is below the threshold.
            penalty = (1 - 0.025 * (cons.sizePenaltyMinRules - self.numAliveRules))
            if penalty <= 0:
                penalty = 0.01
            penalty *= penalty
        if cons.useMDL:
            self.fitness = mdlFit / float(penalty)
        else:
            self.fitness = self.accuracy
            self.fitness *= self.fitness
            self.fitness *= penalty
        self.isEvaluated = True
        
    def getAccuracy(self):
        """ Returns accuracy """
        return self.accuracy
        
    def getActivationsOfRule(self, rule):
        """ Get number of times a given rule has been activated. """
        return self.utilRules[rule]

    def controlBloatRuleDeletion(self):
        """ Determine which rules are to be deleted. """
        nRules = len(self.utilRules)  # Number of rules in agent
        minRules = cons.ruleDeletionMinRules  # Minimum number of rules required
        rulesToDelete = []
        countDeleted = 0
        if nRules > minRules:  # Delete rules only if there are more than the minimum size for an agent.
            for i in range(nRules):
                if self.utilRules[i] == 0:
                    rulesToDelete.append(i)
                    countDeleted += 1
            if (nRules - countDeleted) < minRules:  # If deletion drops the nRules below the min
                rulesToKeep = minRules - (nRules - countDeleted)  # How many of the 'to be deleted rules' should be salvaged to maintain the min.
                for i in range(rulesToKeep):
                    pos = random.randint(0, countDeleted - 1)  # Pick a random rule to save.
                    rulesToDelete.pop(pos)
                    countDeleted -= 1
        return rulesToDelete

    def getFitness(self):
        """ Returns fitness """
        return self.fitness
    
    def getNumAliveRules(self):
        """ Returns the number of Alive rules """
        return self.numAliveRules 

    def getIsEvaluated(self):
        """ Returns whether this rule has already been evaluated. """
        return self.isEvaluated

    def setIsEvaluated(self, eval):
        """ Set whether this rule has now been evaluated. """
        self.isEvaluated = eval
