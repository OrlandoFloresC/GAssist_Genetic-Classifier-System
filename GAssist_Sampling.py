# -*- coding: utf-8 -*-
#!/usr/bin/env python3
#Name: GAssist_Sampling.py

# Import modules
import random

class Sampling:
    def __init__(self, maxSize):
        """ Initialize sampling objects """
        self.maxSize = maxSize
        self.sample = []
        self.num = 0
        self.initSampling()

    def initSampling(self):
        """ Initialize the sampling indexing list and maxSize parameter. """
        self.sample = [i for i in range(self.maxSize)]
        self.num = self.maxSize

    def numSamplesLeft(self):
        """ Returns the number of samples left to draw from. """
        return self.num

    def getSample(self):
        """ Returns the next samples list position, and then removes that position from future consideration. """
        pos = random.randint(0, self.num - 1)
        value = self.sample[pos]
        self.sample[pos] = self.sample[self.num - 1]
        self.num -= 1
        if self.num == 0:
            self.initSampling()
        return value

