from __future__ import absolute_import, division, print_function
import cplex
from Constants import Constants
#from sets import Set
import numpy as np
from sklearn import datasets, linear_model


class SDDPMLCut( SDDPCut ):

    def __init__(self, owner=None, forwardstage=None, trial=-1):
        self.CoefficientQuantityVariable = [[0 for p in self.Instance.ProductSet] for t in self.Instance.TimeBucketSet]
        self.CoefficientProductionVariable = [[0 for p in self.Instance.ProductSet] for t in
                                              self.Instance.TimeBucketSet]
        self.CoefficientStockVariable = [[0 for p in self.Instance.ProductSet] for t in self.Instance.TimeBucketSet]
        self.CoefficientBackorderyVariable = [[0 for p in self.Instance.ProductSet] for t in
                                              self.Instance.TimeBucketSet]

        self.PreviousSolution = None



    def GetPreviousSolutionValue(self, solution):


    def ComputeRegression(self):
        regr = linear