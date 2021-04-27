from __future__ import absolute_import, division, print_function
import cplex
from Constants import Constants
#from sets import Set
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt


#This class contains an attempt to learn the cuts through linear regression.
#However, the method did not provide good results according to our preliminary tests
#Could be deleted.
class SDDPMLCut( object ):

    def __init__(self, owner):
        self.ForwardStage = owner
        self.Instance = self.ForwardStage.SDDPOwner.Instance
        self.QtyConsideredPeriod = [[self.ForwardStage.DecisionStage - t for t in range(self.Instance.LeadTimes[p]) if
                                self.ForwardStage.DecisionStage - t >= 0] for p in self.Instance.ProductSet]
        self.CoefficientQuantityVariable = [[0 for t in self.QtyConsideredPeriod[p]] for p in self.Instance.ProductSet]
        self.CoefficientProductionVariable = [[0 for p in self.Instance.ProductSet] for t in self.Instance.TimeBucketSet]
        self.CoefficientStockVariable = [0 for p in self.Instance.ProductSet]
        self.CoefficientBackorderyVariable = [0 for p in self.Instance.ProductWithExternalDemand]

        self.PreviousSolution = []
        self.PreviousCost = []




    def SavePreviousSolutionValue(self, scenario, AvgCost):


        if Constants.GenerateStrongCut:
            raise ValueError('GenerateStrongCut is True: The data are generated based on the core point')

        values = [self.ForwardStage.SDDPOwner.GetSetupFixedEarlier(p, t, 0) for t in self.Instance.TimeBucketSet for p in self.Instance.ProductSet ] + \
                 [self.ForwardStage.SDDPOwner.GetQuantityFixedEarlier(p, t, scenario) for p in self.Instance.ProductSet for t in self.QtyConsideredPeriod[p]] + \
                 [self.ForwardStage.SDDPOwner.GetInventoryFixedEarlier(p, self.ForwardStage.DecisionStage, scenario) for p in self.Instance.ProductSet] + \
                 [self.ForwardStage.SDDPOwner.GetBackorderFixedEarlier(p, self.ForwardStage.DecisionStage, scenario) for p in self.Instance.ProductWithExternalDemand] ;



        self.PreviousSolution.append(values)
        self.PreviousCost.append(AvgCost)





    def ComputeRegression(self):
        self.Regr = linear_model.LinearRegression()
        self.Regr.fit(self.PreviousSolution, self.PreviousCost)

        print("regression coefficent")
        print(self.Regr.coef_)
        print("regression intercept")
        print(self.Regr.intercept_)


        i = 0
        for t in self.Instance.TimeBucketSet:
            for p in self.Instance.ProductSet:
                self.CoefficientProductionVariable[t][p] = self.Regr.coef_[i]
                self.PlotData(i, self.Regr.coef_[i], self.Regr.intercept_, "Setup_%r_%r"%(p,t))
                i = i+1

        for p in self.Instance.ProductSet:
            for t in range(len(self.QtyConsideredPeriod[p])):
                self.CoefficientQuantityVariable[p][t] = self.Regr.coef_[i]
                self.PlotData(i, self.Regr.coef_[i], self.Regr.intercept_, "Qty_%r_%r"%(p,t))
                i = i + 1

        for p in self.Instance.ProductSet:
                self.CoefficientStockVariable[p] = self.Regr.coef_[i]
                self.PlotData(i, self.Regr.coef_[i], self.Regr.intercept_, "Inventory_%r"%(p))
                i = i + 1

        for p in self.Instance.ProductWithExternalDemand:
                self.CoefficientBackorderyVariable[self.Instance.ProductWithExternalDemandIndex[p]] = self.Regr.coef_[i]
                self.PlotData(i, self.Regr.coef_[i], self.Regr.intercept_, "Backorder_%r"%(p))
                i = i + 1



    def PlotData(self, variable, coeff, intercept, variablename):
       if False and len(self.PreviousSolution) >= 50:

               # self.PreviousSolution = self.PreviousSolution[0:50]
               # self.PreviousCost = self.PreviousCost[0:50]
                data = [d[variable] for d in self.PreviousSolution]
                print(len(data))
                print(len(self.PreviousCost))
                plt.figure(figsize=(4, 4))
                ax = plt.axes()
                ax.scatter(data, self.PreviousCost, color='black', marker='^')

                x = np.linspace(min(data), max(data), 10)

                ax.plot(x, x*coeff + intercept, linestyle='solid', color='blue')

                ax.set_xlabel(variablename + "(%r)"%variable)
                ax.set_ylabel('cost')



                ax.axis('tight')

                plt.show()
