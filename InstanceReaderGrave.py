from __future__ import absolute_import, division, print_function
import numpy as np
import openpyxl as opxl
from ScenarioTreeNode import ScenarioTreeNode
from InstanceReader import InstanceReader
from Tool import Tool
import math
from Constants import Constants
import scipy as scipy

class InstanceReaderGrave(InstanceReader):

    # Constructor
    def __init__(self, instance):
        InstanceReader.__init__(self, instance)
        self.Supplychaindf = None
        self.Datasheetdf = None
        self.Actualdepdemand = [[]]
        self.ActualAvgdemand =[]
        self.Actualstd = [[]]

    def ReadProductList(self):
        self.Instance.ProductName = [] #self.Datasheetdf.Index#[row[0] for row in self.DTFile]
        for row in self.Datasheetdf.index:
            self.Instance.ProductName.append(row)

    #Create datasets from the sheets for instance from Grave 2008
    def OpenFiles(self, instancename):
        wb2 = opxl.load_workbook("./Instances/GraveFiles/MSOM-06-038-R2.xlsx")
        # The supplychain is defined in the sheet named "01_LL" and the data are in the sheet "01_SD"
        self.Supplychaindf = Tool.ReadDataFrame(wb2, instancename + "_LL")
        self.Datasheetdf = Tool.ReadDataFrame(wb2, instancename + "_SD")
        self.Datasheetdf = self.Datasheetdf.fillna(0)

    def ReadNrResource(self):
        self.Instance.NrResource = len(self.Instance.ProductName)

    # Compute the requireement from the supply chain. This set of instances assume the requirement of each arc is 1.
    def CreateRequirement(self):
        self.Instance.Requirements = [[0] * self.Instance.NrProduct for _ in self.Instance.ProductSet]
        for i, row in self.Supplychaindf.iterrows():
            self.Instance.Requirements[self.Instance.ProductName.index(row.get_value('destinationStage'))][self.Instance.ProductName.index(i)] = 1

    def GetEchelonHoldingCost(self, uselessparameter):
         result = [(0.1 / 250) * self.Datasheetdf.get_value(self.Instance.ProductName[p], 'stageCost') for p in self.Instance.ProductSet]
         return result

    def GetProductLevel(self):
         result = [self.Datasheetdf.get_value(self.Instance.ProductName[p], 'relDepth') for p in self.Instance.ProductSet]
         return result

    def GenerateTimeHorizon(self, largetimehorizon = False, largetimehorizonperiod = 10, additionaltimehorizon = 0):
        # Consider a time horizon of 20 days plus the total lead time
        self.Instance.NrTimeBucket = (2 * self.Instance.MaxLeadTime) + additionaltimehorizon

        if largetimehorizon:
            self.Instance.NrTimeBucket = largetimehorizonperiod
        self.Instance.NrTimeBucketWithoutUncertaintyBefore = self.Instance.MaxLeadTime
        self.Instance.NrTimeBucketWithoutUncertaintyAfter = 0
        self.Instance.ComputeIndices()

    def GenerateDistribution(self, forecasterror, rateknown, longtimehorizon = False):
        # Generate the sets of scenarios
        self.Instance.YearlyAverageDemand = [self.Datasheetdf.get_value(self.Instance.ProductName[p], 'avgDemand')
                                              for p in self.Instance.ProductSet]


        self.Instance.YearlyStandardDevDemands = [self.Datasheetdf.get_value(self.Instance.ProductName[p], 'stdDevDemand')
                                                    for p in self.Instance.ProductSet]
        #
        # if self.Instance.Distribution == Constants.SlowMoving:
        #     self.Instance.YearlyAverageDemand = [1 if self.Datasheetdf.get_value(self.Instance.ProductName[p], 'avgDemand') > 0
        #                                           else 0
        #                                         for p in self.Instance.ProductSet]
        #     self.Instance.YearlyStandardDevDemands = [1 if self.Datasheetdf.get_value(self.Instance.ProductName[p], 'avgDemand') > 0
        #                                           else 0
        #                                         for p in self.Instance.ProductSet]
        #
        # if self.Instance.Distribution == Constants.Uniform:
        #     self.Instance.YearlyAverageDemand = [0.5 if self.Datasheetdf.get_value(self.Instance.ProductName[p], 'avgDemand') > 0
        #                                          else 0
        #                                          for p in self.Instance.ProductSet]


        stationarydistribution = self.IsStationnaryDistribution()

        if stationarydistribution:
            self.GenerateStationaryDistribution()

        else:

            self.GenerateNonStationary()
            self.Instance.ForecastError = [forecasterror for p in self.Instance.ProductSet]
            self.Instance.RateOfKnownDemand = [math.pow(rateknown, t + 1) for t in self.Instance.TimeBucketSet]
            self.Instance.ForecastedAverageDemand = [[np.floor(np.random.normal(self.Instance.YearlyAverageDemand[p],
                                                                                 self.Instance.YearlyStandardDevDemands[p], 1).clip(min=0.0)).tolist()[0]
                                                      if self.Instance.YearlyStandardDevDemands[p] > 0
                                                      else float(self.Instance.YearlyAverageDemand[p])
                                                      for p in self.Instance.ProductSet] for t in self.Instance.TimeBucketSet]

            self.Instance.ForcastedStandardDeviation = [[(1 - self.Instance.RateOfKnownDemand[t])
                                                           * self.Instance.ForecastError[p]
                                                           * self.Instance.ForecastedAverageDemand[t][p]
                                                           if t < (self.Instance.NrTimeBucket - self.Instance.NrTimeBucketWithoutUncertaintyAfter)
                                                           else 0.0
                                                            for p in self.Instance.ProductSet]
                                                         for t in self.Instance.TimeBucketSet]






    def GenerateNonStationary(self, forecasterror, rateknown):
        self.Instance.ForecastError = [forecasterror for p in self.Instance.ProductSet]
        self.Instance.RateOfKnownDemand = [
            math.pow(rateknown, (t - self.Instance.NrTimeBucketWithoutUncertaintyBefore + 1))
            for t in self.Instance.TimeBucketSet]
        self.Instance.ForecastedAverageDemand = [[0.0 for p in self.Instance.ProductSet]
                                                 for t in self.Instance.TimeBucketSet]

        prodindex = 0
        finishproduct = self.GetfinishProduct()
        for p in range(len(finishproduct)):
                prodindex = finishproduct[p]
                timeindex = 0
                stochastictime = range(self.Instance.NrTimeBucketWithoutUncertaintyBefore,
                                       self.Instance.NrTimeBucket - self.Instance.NrTimeBucketWithoutUncertaintyAfter)
                for t in stochastictime:
                    timeindex += 1

                    if t <> self.Instance.NrTimeBucketWithoutUncertaintyBefore + int(self.DTFile[timeindex][0]) - 1:
                        raise NameError("Wrong time %d - %d -%d" % (t, int(self.DTFile[timeindex][0]) - 1, timeindex))

                    self.Instance.ForecastedAverageDemand[t][prodindex] = float(self.DTFile[timeindex][p + 1])

                for t in range(self.Instance.NrTimeBucket - self.Instance.NrTimeBucketWithoutUncertaintyAfter,
                               self.Instance.NrTimeBucket):
                    self.Instance.ForecastedAverageDemand[t][prodindex] = sum(
                        self.Instance.ForecastedAverageDemand[t2][prodindex]
                        for t2 in stochastictime) / len(stochastictime)

                self.Instance.YearlyAverageDemand = [sum(self.Instance.ForecastedAverageDemand[t][p]
                                                         for t in self.Instance.TimeBucketSet
                                                         if t >= self.Instance.NrTimeBucketWithoutUncertaintyBefore)
                                                     / (
                                                                 self.Instance.NrTimeBucket - self.Instance.NrTimeBucketWithoutUncertaintyBefore)
                                                     for p in self.Instance.ProductSet]

        self.Instance.ForcastedStandardDeviation = [[(1 - self.Instance.RateOfKnownDemand[t])
                                                     * self.Instance.ForecastError[p]
                                                     * self.Instance.ForecastedAverageDemand[t][p]
                                                     if t < (
                    self.Instance.NrTimeBucket - self.Instance.NrTimeBucketWithoutUncertaintyAfter)
                                                     else 0.0
                                                     for p in self.Instance.ProductSet]
                                                    for t in self.Instance.TimeBucketSet]

        self.Instance.YearlyStandardDevDemands = [sum(self.Instance.ForcastedStandardDeviation[t][p]
                                                      for t in self.Instance.TimeBucketSet) / self.Instance.NrTimeBucket
                                                  for p in self.Instance.ProductSet]



    #This function generate the starting inventory
    def GenerateStartinInventory(self):
        self.Instance.StartingInventories = [0.0 for p in self.Instance.ProductSet]

    def GenerateSetup(self, echelonstocktype):
        # Assume a starting inventory is the average demand during the lead time
        echeloninventorycost = self.GetEchelonHoldingCost(echelonstocktype)

        self.Instance.SetupCosts = [(self.DependentAverageDemand[p]
                              * echeloninventorycost[p]
                              * 0.5
                              * (self.TimeBetweenOrder) * (self.TimeBetweenOrder))
                           for p in self.Instance.ProductSet]

    def GenerateCapacity(self, capacityfactor):
        self.Instance.NrResource = self.Instance.NrLevel
        self.Instance.ProcessingTime = [[self.Datasheetdf.get_value(self.Instance.ProductName[p], 'stageTime')
                                            if (self.Level[p] == k) else 0.0

                                            for k in range(self.Instance.NrResource)]
                                           for p in self.Instance.ProductSet]

        self.Instance.Capacity = [capacityfactor * sum(self.DependentAverageDemand[p] * self.Instance.ProcessingTime[p][k]
                                                         for p in self.Instance.ProductSet)
                                   for k in range(self.Instance.NrResource)]

    def GenerateTransportCost(self, alternatetype):
        self.Instance.AternateCosts = [[0] * self.Instance.NrProduct for _ in self.Instance.ProductSet]
        self.Instance.Alternates = [[0] * self.Instance.NrProduct for _ in self.Instance.ProductSet]

        for p in self.Instance.ProductSet:
            if self.Datasheetdf.get_value(self.Instance.ProductName[p], 'stageClassification') == "Retail":

                maincomponentfound = False
                for q in self.Instance.ProductSet:
                    if self.Instance.Requirements[p][q]:
                        if not maincomponentfound:
                            maincomponentfound = True
                            maincomponent = q
                        else:
                            self.Instance.Requirements[p][q] = 0
                            self.Instance.Alternates[maincomponent][q] = 1

