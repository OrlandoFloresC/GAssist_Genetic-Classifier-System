# -*- coding: utf-8 -*-
#!/usr/bin/env python3
#Name: GAssist_pop.py

# Import modules
from GAssist_Constants import *  
from GAssist_Agent import *
from GAssist_Sampling import *
from GAssist_Global_MDL import *
from random import *
from copy import deepcopy
import math
#import torch

class GAssistPop:    
    def __init__(self, e, inSet, testSet, w, smart, cw, maxProblems):
        print("Initializing Population")
        self.env = e
        self.windows = w
        self.maxProblems = maxProblems
        self.tempPop = []
        self.pop = []  # The population of Pitts individuals

        # Data/ Instance Variables
        self.instances = inSet
        self.testSet = testSet
        
        # Windowing/Version Handling
        self.numVersions = cons.numStrata
        self.bestAgents = [None for _ in range(self.numVersions)]  # A list of agent/individual objects representing the best found for each version/window.
        
        # Initialize Population of Agents
        print(f"Making {cons.popSize} agents.")
        self.pop = [GAssistAgent(self.env, smart, cw) for _ in range(cons.popSize)]

        print("Population Initialization Complete.")
        
        # Tracking Statistics
        self.trackBestFitness = 0.0
        self.bestFitness = 0.0
        self.bestAccuracy = 0.0
        self.bestNumRules = 0.0
        self.bestAliverules = 0.0
        self.bestGenerality = 0.0
        
        self.averageFitness = 0.0
        self.averageAccuracy = 0.0
        self.averageNumRules = 0.0
        self.averageNumRulesUtils = 0.0
        self.averageGenerality = 0.0
        
        self.last10Accuracy = [0.0 for _ in range(10)]
        self.last10IterationsAccuracyAverage = 0.0
        self.iterationsSinceBest = 0
        self.countStatistics = 0
        self.firstIteration = True
        
        self.iterationNichingDisabled = None

        self.globalMDL = GAssistGlobalMDL(self.env)  # Initializes this class for MDL calculations.

        print("Calculating initial performance.")
        self.doEvaluation(self.pop)  # Done using whole training set.
        self.calculateStatistics()

    def checkBestIndividual(self):
        """ Identifies and sets the best agent for this learning iteration. - Responsible for updating bestIndiv[]. """
        best = self.getBest()
        currVer = self.windows.getCurrentVersion()
        if self.bestAgents[currVer] is None or best.compareToIndividual(self.bestAgents[currVer]):
            self.bestAgents[currVer] = best.clone()
        
    def getPopClone(self):
        """ Returns a population clone so that evaluations can be called from GAssist module non destructively. """
        return deepcopy(self)
    
    def doEvaluation(self, pop):
        # Evaluates the performance of each agent in the population.
        for i in range(cons.popSize):
            if not pop[i].agnPer.isEvaluated:
                self.evaluateClassifier(pop[i])


    def evaluateClassifier(self, ind):
        # Evaluates the performance of each agent in the population.
        instanceWindow = self.windows.getInstances()
        ind.agnPer.resetPerformance(ind.getNumRules())
        for instance in instanceWindow:
            real = instance[1] 
            predicted = ind.classify(instance)  
            ind.agnPer.addPrediction(predicted, real)
        
        ind.agnPer.calculatePerformance(self.globalMDL, ind)
        
        if cons.doRuleDeletion:
            ind.deleteRules(ind.agnPer.controlBloatRuleDeletion())
    
        
    def calculateStatistics(self):
        # Calculates the statistics of the current agent population 
        popLength = cons.popSize
        sumFitness = sumAccuracy = sumNumRules = sumNumRulesUtils = sumGenerality = 0.0
        
        for agent in self.pop:
            sumFitness += agent.agnPer.getFitness()
            sumAccuracy += agent.agnPer.getAccuracy()
            sumNumRules += agent.getRealNumRules()
            sumNumRulesUtils += agent.agnPer.getNumAliveRules()
            sumGenerality += agent.getGenerality()

        self.averageFitness = sumFitness / popLength
        self.averageAccuracy = sumAccuracy / popLength
        self.averageNumRules = sumNumRules / popLength
        self.averageNumRulesUtils = sumNumRulesUtils / popLength
        self.averageGenerality = sumGenerality / popLength

        # Get the Best info
        bestAgent = self.getBest()

        self.bestFitness = bestAgent.agnPer.getFitness()
        self.bestAccuracy = bestAgent.agnPer.getAccuracy()
        self.bestNumRules = bestAgent.getRealNumRules()
        self.bestAliverules = bestAgent.agnPer.getNumAliveRules()
        self.bestGenerality = bestAgent.getGenerality()

        self.last10Accuracy.pop(0)
        self.last10Accuracy.append(self.bestAccuracy)

        self.bestOfIteration(bestAgent.agnPer.getFitness())
        self.countStatistics += 1 
    

    def bestOfIteration(self, itBestFit):
        """ Determines the best agent for the current iteration. """
        if self.iterationsSinceBest == 0:
            self.iterationsSinceBest += 1
            if self.firstIteration:
                self.trackBestFitness = itBestFit
                self.firstIteration = False
                print("Set initial best fitness.")
        else: 
            newBest = False
            if cons.useMDL:
                if itBestFit < self.trackBestFitness:
                    newBest = True
            else:
                if itBestFit > self.trackBestFitness:
                    newBest = True
            if newBest:
                print("NEW BEST AGENT FOUND")
                self.trackBestFitness = itBestFit
                self.iterationsSinceBest = 1
            else:
                self.iterationsSinceBest += 1
        
        i = max(0, self.countStatistics - 9)
        max_iter = self.countStatistics + 1 
        num = max_iter - i
        self.last10IterationsAccuracyAverage = sum(self.last10Accuracy[i:max_iter]) / float(num)
        
                
    def getBest(self):
        """ Returns the best agent in the population. """
        posWinner = 0
        for i in range(len(self.pop)):
            if self.pop[i].compareToIndividual(self.pop[posWinner]):
                posWinner = i
        return self.pop[posWinner]

    def getWorst(self, offspring):
        """ Returns the location of the worst agent in the population. """
        posWorst = 0
        for i in range(len(offspring)):
            if not offspring[i].compareToIndividual(offspring[posWorst]):
                posWorst = i
        return posWorst

    def resetBestStats(self):
        """ Reinitializes the iterations since best. """
        self.iterationsSinceBest = 0

    def evolvePopulation(self, lastIteration, iteration):
        """ Directs the evolution of the agent population. Calls the major operations. """
        res1 = self.windows.newIteration()  # Shifts data set portion to new window. Everything done during this evolution is from the perspective of this window (including best agent)
        res2 = self.runTimers(iteration)

        if res1 or res2:
            self.setModified()  # isEvaluated always reset for all agents when window switches

        # GA Cycle - Main learning operations
        self.pop = self.doTournamentSelection()  # TournamentSelects the population to make a new population of best agents.
        offspring = self.doNicheCrossover() if cons.defaultClass != "disabled" else self.doCrossover()  # new population object
        self.doMutation(offspring)
        self.doEvaluation(offspring)
        self.pop = self.replacementPolicy(offspring, lastIteration)
        self.checkBestIndividual()
        self.calculateStatistics()
        self.checkNichingStatus(iteration)

    def trainPopulation(self):
        """ Handles the full training evaluation - not repetitive because final replacement policy is implemented here. """
        self.tempPop = deepcopy(self.pop)
        self.setModified()
        self.pop = self.replacementPolicy(self.pop, True)
        self.doEvaluation(self.pop)
        self.calculateStatistics()

    def returnPopulation(self):
        """ Reverts to original population from before train/test evaluation. """
        self.pop = self.tempPop

    def runTimers(self, iteration):
        """ Perform the timer functions """
        res1 = self.globalMDL.newIteration(iteration, self)
        res2 = self.timerBloat(iteration)

        return res1 or res2

    def timerBloat(self, iteration):
        """ Resets some new bloat constants at timed checkpoints. """
        cons.setRuleDelete(iteration >= cons.iterationRuleDeletion)
        cons.setHierarchy(iteration >= cons.iterationHierarchicalSelection)

        if iteration == self.maxProblems - 1:
            cons.setDeleteMinRules(1)
            return True

        if iteration == cons.iterationRuleDeletion:
            return True

        return False



    def doTournamentSelection(self):
        """ Does Tournament Selection without replacement """
        selectedPopulation = [None for _ in range(cons.popSize)]
        numNiches = self.env.getNrActions() if cons.nichingEnabled else 1  # Determine the number of niches

        pools = [[] for _ in range(numNiches)]  # will hold lists of agent IDs
        nicheCounters = [int(cons.popSize / numNiches) for _ in range(numNiches)]  # Balanced number of times that selection occurs in each niche

        for i in range(cons.popSize):  # For each agent in the population
            niche = self.selectNicheWOR(nicheCounters)  # Selects a niche from which to perform selection
            winner = self.selectCandidateWOR(pools[niche], niche)  # Selects a single agent ID from given niche
            for _ in range(cons.tournamentSize):  # Do tournament selection
                candidate = self.selectCandidateWOR(pools[niche], niche)  # Pick another agent ID from given niche
                if self.pop[candidate].compareToIndividual(self.pop[winner]):
                    winner = candidate
            selectedPopulation[i] = self.pop[winner].clone()  # Implement copy() - base on clone.

        return selectedPopulation  # Same array format as self.pop

    def selectNicheWOR(self, quotas):
        """ Selects a niche without replacement. """
        num = len(quotas)  # Number of niches
        if num == 1:  # If niching is turned off
            return 0

        total = sum(quotas)
        if total == 0:  # Catch for the case that the initial total is less than popSize.
            return randint(0, num - 1)  # Returns random niche ID.

        pos = randint(0, total - 1)  # Pick a random agent ID.
        total = 0
        for i in range(num):
            total += quotas[i]
            if pos < total:
                quotas[i] -= 1
                return i  # Returns niche ID

        print("Error - selectNicheWOR")
        return -1

    def selectCandidateWOR(self, pool, whichNiche):
        """ Selects a candidate without replacement. """
        if len(pool) == 0:  # Build the pool
            self.initPool(pool, whichNiche)
            if len(pool) == 0:  # If pool is still empty - return a random agent from the whole population.
                return randint(0, cons.popSize - 1)

        pos = randint(0, len(pool) - 1)  # Pick an agent ID from this pool
        elem = int(pool[pos])
        pool.pop(pos)
        return elem

    def initPool(self, pool, whichNiche):
        """ Builds the pool list for the specified niche. Pool list is an array of agent ID references. """
        if cons.nichingEnabled:
            for i in range(cons.popSize):
                if self.pop[i].getNiche() == whichNiche:  # If the agent is part of the selected niche
                    pool.append(i)
        else:
            for i in range(cons.popSize):
                pool.append(i)

        

    def checkNichingStatus(self, iteration):
        """ Checks whether niching is still needed for selection - Niching is turned off once the tournament with 
    niche preservation is used until the best individuals of each default class have similar training accuracy. """
    
        if cons.nichingEnabled:
            numNiches = self.env.getNrActions()
            counters = [0 for i in range(numNiches)]
            nicheFit = [0.0 for i in range(numNiches)]
            for i in range(len(self.pop)):
                niche = self.pop[i].getNiche()
                counters[niche] += 1
                indAcc = self.pop[i].agnPer.getAccuracy()
                if indAcc > nicheFit[niche]:
                    nicheFit[niche] = indAcc
            
            if len(cons.accDefaultRules[0]) == 15:  # If there have been at least 15 iterations.
                for i in range(numNiches):
                    cons.accDefaultRules[i].pop(0)  # remove oldest accuracy
            
            for i in range(numNiches):
                cons.accDefaultRules[i].append(nicheFit[i])

            if len(cons.accDefaultRules[0]) == 15:
                aves = []
                for i in range(numNiches):
                    aveN = self.getAverage(cons.accDefaultRules[i])
                    aves.append(aveN)
        
                dev = self.getDeviation(aves)
                if dev < 0.005:
                    print(f"Iteration {iteration}, niching disabled")
                    cons.setNichingStatus(False)
                    self.iterationNichingDisabled = iteration


    def doNicheCrossover(self):
        """ Arranges the crossover within each niche and selects parents. Does not do actual crossing over of agents. """
        countCross = 0
        numNiches = self.env.getNrActions()
        parents = [[] for i in range(numNiches)]  # separates potential parents(agents) into groups based on niche.
        parent1 = None
        parent2 = None
        offspring = [None, None]
        offspringPopulation = [[] for i in range(cons.popSize)]
        
        for i in range(cons.popSize):  # construct parents
            niche = self.pop[i].getNiche()
            parents[niche].append(i)
            
        for i in range(numNiches):  # do crossover separately for each niche
            size = len(parents[i])  # number of parents in this niche
            samp = Sampling(size)
            p1 = -1
            for j in range(size):  # for each parent.
                if random() < cons.probCrossover:
                    if p1 == -1:  # gets a first parent and takes up a step when a crossover was successful
                        p1 = samp.getSample()
                    else:
                        p2 = samp.getSample()
                        pos1 = parents[i][p1]
                        pos2 = parents[i][p2]
                        parent1 = self.pop[pos1]
                        parent2 = self.pop[pos2]
                        
                        offspring = parent1.crossoverClassifiers(parent2)
                        offspringPopulation[countCross] = offspring[0]
                        countCross += 1
                        offspringPopulation[countCross] = offspring[1]
                        countCross += 1
                        p1 = -1
                else:
                    pos = parents[i][samp.getSample()]
                    offspringPopulation[countCross] = self.pop[pos].clone()
                    countCross += 1
            if p1 != -1:
                pos = parents[i][p1]
                offspringPopulation[countCross] = self.pop[pos].clone()
                countCross += 1

        return offspringPopulation


    def doCrossover(self):
        """ Arranges the crossover within the entire population and selects parents. Does not do actual crossing over of agents. """
        countCross = 0
        parents = []
        parent1 = None
        parent2 = None
        offspring = [None, None]
        offspringPopulation = [[] for i in range(cons.popSize)]
        
        for i in range(cons.popSize):  # construct parent position list
            parents.append(i)
            
        p1 = -1
        samp = Sampling(cons.popSize)
        for i in range(cons.popSize):  # for each parent.
            if random() < cons.probCrossover:
                if p1 == -1:  # gets a first parent
                    p1 = samp.getSample()
                else:
                    p2 = samp.getSample()
                    pos1 = parents[p1]
                    pos2 = parents[p2]
                    parent1 = self.pop[pos1]
                    parent2 = self.pop[pos2]
                        
                    offspring = parent1.crossoverClassifiers(parent2)
                    offspringPopulation[countCross] = offspring[0]
                    countCross += 1
                    offspringPopulation[countCross] = offspring[1]
                    countCross += 1
                    p1 = -1
            else:
                pos = parents[samp.getSample()]
                offspringPopulation[countCross] = self.pop[pos].clone()
                countCross += 1
        if p1 != -1:
            pos = parents[p1]
            offspringPopulation[countCross] = self.pop[pos].clone()
            countCross += 1
                
        return offspringPopulation

    
    
    def doMutation(self, offspring):
        # Just calls to mutate population of agents in GAssist_Agent. Also initiates the special stages.
        for i in range(cons.popSize):
            if random() < cons.probMutationInd:  # decides if a whole agent will undergo mutation.
                offspring[i].doMutation()
    

    def setModified(self):
        """ Set entire population to not evaluated. """
        for i in range(cons.popSize):
            self.pop[i].agnPer.setIsEvaluated(False)

    def replacementPolicy(self, offspring, lastIteration):
        """ Handles Elitism and replacement of worst rules in the population by the best old ones """
        if lastIteration:  # LAST ITERATION ONLY
            for i in range(self.numVersions):
                if self.bestAgents[i] is not None:
                    self.evaluateClassifier(self.bestAgents[i])  # evaluate each best agent on some other subset of the data?

            set_ = []
            for i in range(cons.popSize):
                self.sortInsert(set_, offspring[i])

            for i in range(self.numVersions):
                if self.bestAgents[i] is not None:
                    self.sortInsert(set_, self.bestAgents[i])

            for i in range(cons.popSize):  # orders the offspring
                offspring[i] = set_[i]

        else:  # ALL OTHER ITERATIONS
            previousVerUsed = False
            currVer = self.windows.getCurrentVersion()  # Returns the current window id. (eg. 0,1,2,3)

            if self.bestAgents[currVer] is None and currVer > 0:  # If there is no best agent and there is at least one current Version so far
                previousVerUsed = True
                currVer -= 1  # use previous window.

            if self.bestAgents[currVer] is not None:  # elitism
                self.evaluateClassifier(self.bestAgents[currVer])
                worst = self.getWorst(offspring)
                offspring[worst] = self.bestAgents[currVer].clone()

            if not previousVerUsed:
                prevVer = 0
                if currVer == 0:
                    prevVer = self.numVersions - 1  # assumes that currVers can go to 0 as one of its versions.
                else:
                    prevVer = currVer - 1

                if self.bestAgents[prevVer] is not None:
                    self.evaluateClassifier(self.bestAgents[prevVer])
                    worst = self.getWorst(offspring)
                    offspring[worst] = self.bestAgents[prevVer].clone()

        return offspring

    def sortInsert(self, set_, cl):
        """ Sorts agents into a list according to comparisons. """
        for i in range(len(set_)):
            if cl.compareToIndividual(set_[i]):
                set_.insert(i, cl)
                return
        set_.append(cl)

    def getTracking(self):
        """ Get all the tracking values packaged as an array. """
        return [
            self.bestFitness,
            self.bestAccuracy,
            self.bestNumRules,
            self.bestAliverules,
            self.bestGenerality,
            self.averageFitness,
            self.averageAccuracy,
            self.averageNumRules,
            self.averageNumRulesUtils,
            self.averageGenerality,
        ]  # 10 items

    def getIterationsSinceBest(self):
        """ Returns iterations since best """
        return self.iterationsSinceBest

    # UTILITIES ************************************************************************************
    def getAverage(self, data):
        """ Get the average of a list of values. """
        ave = sum(data) / float(len(data))
        return ave

    def getDeviation(self, data):
        """ Get the standard deviation from a list of values. """
        ave = self.getAverage(data)
        dev = sum((val - ave) ** 2 for val in data) / len(data)
        return math.sqrt(dev)
