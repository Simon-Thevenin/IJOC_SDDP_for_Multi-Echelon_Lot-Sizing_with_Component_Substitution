from __future__ import absolute_import, division, print_function
from Constants import Constants
from random import randint
import math


class InstanceReader(object):


    # Constructor
    def __init__(self, instance):
        # The instance currently built
        self.Instance = instance
        self.DependentAverageDemand = []
        self.TimeBetweenOrder = 2
        self.InstanceType = ""
        self.Level = []  # indicates the level of each produce
        self.LevelSet = []
        self.Filename = ""

    # This function read the instance from the file (it is the main function in this class)
    # b and lostsale are the cost structure (backlog cost are b*instentory costs, lost sale costs are lostsale*instentory costs
    # is the echelon inventory cost (n: normal, l: large increase at each echelon)
    # forecasterror and rateknown are parameter used to build the NonStationary distribution
    def ReadFromFile(self, instancename, distribution="NonStationary", longtimehoizon=False, largetimehorizonperiod = 10, alternatetype = "Normal"):
            backlogcostmultiplier = 2
            forcasterror = 25
            e = "n"
            rateknown = 50
            leadtimestructure = 0
            lostsalecostmultiplier = 20
            capacityfactor = 2

            self.Instance.InstanceName = "%s_%s_b%s_fe%s_e%s_rk%s_ll%s_l%s_H%s%s_c%s" % (
                                                    instancename, distribution, backlogcostmultiplier, forcasterror,
                                                    e, rateknown, leadtimestructure, lostsalecostmultiplier,
                                                    longtimehoizon, largetimehorizonperiod, capacityfactor)
            self.Instance.Distribution = distribution
            self.Filename = instancename
            # Open the file, and read the product, requirement, and resources
            self.OpenFiles(instancename)
            self.ReadProductList()
            self.ReadInstanceStructure()
            self.ReadNrResource()
            self.CreateRequirement()

            self.GenerateTransportCost(alternatetype)
            self.Instance.ComputeLevel()
            self.CreateLeadTime(leadtimestructure)
            self.GenerateHoldingCostCost(e)
            self.Instance.ComputeMaxLeadTime()
            self.GenerateTimeHorizon(longtimehoizon, largetimehorizonperiod=largetimehorizonperiod)
            self.GenerateDistribution(float(forcasterror / 100.0), float(rateknown / 100.0),
                                      longtimehorizon=longtimehoizon)
            self.ComputeAverageDependentDemand()
            self.GenerateStartinInventory()
            self.GenerateSetup(e)
            self.GenerateCapacity(capacityfactor)
            self.GenerateCostParameters(backlogcostmultiplier, lostsalecostmultiplier)
            self.GenerateVariableCost()
            self.Instance.SaveCompleteInstanceInExelFile()
            self.Instance.ComputeInstanceData()

    # This function reads the number of products, resources, ...
    def ReadInstanceStructure(self):
            self.Instance.NrResource = len(self.Instance.ProductName)
            self.Instance.NrProduct = len(self.Instance.ProductName)
            self.Instance.NrTimeBucket = 0
            self.Instance.ComputeIndices()

            print( "Can I remove that:" )
            self.Level = []  # indicates the level of each produce
            self.LevelSet = []



    # This function creates the lead times
    # leadtimestructure = 0: all items/alternative have a lead time of 1
    # leadtimestructure = 1: all components have a lead time of 1, end items have a lead time of 0
    # leadtimestructure = 2: the lead time is chosen randomly in [0, 3], but the total time to transform end items to component is lower than 5
    def CreateLeadTime(self, leadtimestructure):
        self.Instance.LeadTimes = [1 for p in self.Instance.ProductSet]

        if leadtimestructure == 1:
            productwith0leadtime = [p for p in self.Instance.ProductSet if self.Instance.Level[p] == 1]
            for p in productwith0leadtime:
               self.Instance.LeadTimes[p] = 0

        if leadtimestructure == 2:
            self.Instance.MaxLeadTime = 10
            while self.Instance.MaxLeadTime > 5:
                self.Instance.LeadTimes = [randint(0, 3) for p in self.Instance.ProductSet]
                self.Instance.ComputeMaxLeadTime()

        # Generate the inventory costs
    def GenerateHoldingCostCost(self, e="n"):
        # Assume an inventory holding cost of 0.1 per day for now
        holdingcost = 1  # 0.1 / 250
        self.Instance.InventoryCosts = [0.0] * self.Instance.NrProduct
        # The cost of the product is given by  added value per stage. The cost of the product at each stage must be computed
        addedvalueatstage = self.GetEchelonHoldingCost(e)
        self.Level = self.GetProductLevel()
        self.LevelSet = sorted(set(self.Level), reverse=True)
        for l in self.LevelSet:
            prodinlevel = [p for p in self.Instance.ProductSet if self.Level[p] == l]
            for p in prodinlevel:
                addedvalueatstage[p] = sum(addedvalueatstage[q] * self.Instance.Requirements[p][q]
                                           for q in self.Instance.ProductSet)\
                                       + addedvalueatstage[p]
                self.Instance.InventoryCosts[p] = holdingcost * addedvalueatstage[p]

        if Constants.Debug:
            print( "Inventory cost:%r" % self.Instance.InventoryCosts )

    def ComputeAverageDependentDemand(self):
        self.Level = self.GetProductLevel()
        self.ActualAvgdemand = [math.ceil(float(sum(self.Instance.ForecastedAverageDemand[t][p]
                                                        for t in self.Instance.TimeBucketSet)) \
                                 / float( self.Instance.NrTimeBucket - self.Instance.NrTimeBucketWithoutUncertaintyBefore))
                                    for p in self.Instance.ProductSet]
        self.Actualdepdemand = [[self.Instance.ForecastedAverageDemand[t][p] for p in self.Instance.ProductSet] for
                                    t in
                                    self.Instance.TimeBucketSet]

        self.DependentAverageDemand = [self.ActualAvgdemand[p] for p in self.Instance.ProductSet]
        self.LevelSet = sorted(set(self.Level), reverse=False)

        for l in self.LevelSet:
            prodinlevel = [p for p in self.Instance.ProductSet if self.Level[p] == l]
            for p in prodinlevel:
                self.DependentAverageDemand[p] = sum(self.DependentAverageDemand[q] * self.Instance.Requirements[q][p]
                                                    for q in self.Instance.ProductSet) \
                                                 + self.DependentAverageDemand[p]
                for t in self.Instance.TimeBucketSet:
                    self.Actualdepdemand[t][p] = sum(self.Actualdepdemand[t][q] * self.Instance.Requirements[q][p]
                                                     for q in self.Instance.ProductSet) \
                                                     + self.Actualdepdemand[t][p]


        self.Actualstd = [[self.Instance.ForcastedStandardDeviation[t][p] for p in self.Instance.ProductSet]
                          for t in self.Instance.TimeBucketSet]
        self.LevelSet = sorted(set(self.Level), reverse=False)
        for l in self.LevelSet:
            prodinlevel = [p for p in self.Instance.ProductSet if self.Level[p] == l]
            for t in self.Instance.TimeBucketSet:
                for p in prodinlevel:
                    self.Actualstd[t][p] = sum(self.Actualstd[t][q]
                                               * self.Instance.Requirements[q][p]
                                               * self.Instance.Requirements[q][p] for q in self.Instance.ProductSet) \
                                               + self.Actualstd[t][p]

    def GenerateSetup(self):
        # Assume a starting inventory is the average demand during the lead time
        echeloninventorycost = [self.Instance.InventoryCosts[p] \
                                    - sum( self.Instance.Requirements[p][q] * self.Instance.InventoryCosts[q]
                                           for q in self.Instance.ProductSet)
                                    for p in self.Instance.ProductSet]


        self.Instance.SetupCosts = [(self.DependentAverageDemand[p]
                                         * echeloninventorycost[p]
                                         * 0.5
                                         * (self.TimeBetweenOrder) * (self.TimeBetweenOrder))
                                        for p in self.Instance.ProductSet]

    def GenerateCapacity(self, capacityfactor):
        raise NameError( "Capacity factor is not used" )
        self.Instance.NrResource = self.Instance.NrLevel
        self.Instance.ProcessingTime = [[self.Datasheetdf.get_value(self.Instance.ProductName[p], 'stageTime')
                                         if (self.Level[p] == k) else 0.0

                                         for k in range(self.Instance.NrResource)]
                                         for p in self.Instance.ProductSet]
        capacityfactor = 2;
        self.Instance.Capacity = [
            capacityfactor * sum(self.DependentAverageDemand[p] * self.Instance.ProcessingTime[p][k]
                                 for p in self.Instance.ProductSet)
            for k in range(self.Instance.NrResource)]

    def GenerateCostParameters(self, b, lostsale):
        # Gamma is set to 0.9 which is a common value (find reference!!!)
        self.Instance.Gamma = 1.0
        # Back order is twice the  holding cost as in :
        # Solving the capacitated lot - sizing problem with backorder consideration CH Cheng1 *, MS Madan2, Y Gupta3 and S So4
        # See how to set this value
        self.Instance.BackorderCosts = [b * self.Instance.InventoryCosts[p] for p in self.Instance.ProductSet]
        self.Instance.LostSaleCost = [lostsale * self.Instance.InventoryCosts[p] for p in self.Instance.ProductSet]

    def GenerateVariableCost(self):
        self.Instance.VariableCost = [sum(self.Instance.Requirements[p][q] * self.Instance.InventoryCosts[q]
                                          for q in self.Instance.ProductSet)
                                          for p in self.Instance.ProductSet]

    #return the set of end-item
    def GetfinishProduct(self):
            finishproduct = []
            for p in self.Instance.ProductSet:
                nrrequired = sum(1 for q in self.Instance.ProductSet if self.Instance.Requirements[q][p])
                nralternate = sum(1 for q in self.Instance.ProductSet if self.Instance.Alternates[q][p])
                if nrrequired == 0 and nralternate == 0:
                    finishproduct.append(p)

            return finishproduct


    #Return true if the considered distribution is stationary
    def IsStationnaryDistribution(self):
            stationarydistribution = (self.Instance.Distribution == Constants.SlowMoving) \
                                     or (self.Instance.Distribution == Constants.Lumpy) \
                                     or (self.Instance.Distribution == Constants.Binomial)
            return stationarydistribution


    #This function generate the distribution Lumpy, Slowmoving or binomial
    def GenerateStationaryDistribution(self):

            finishproduct = self.GetfinishProduct()

            # Generate the sets of scenarios
            if self.Instance.Distribution == Constants.SlowMoving:
                self.Instance.YearlyAverageDemand = [1 if p in finishproduct else 0 for p in self.Instance.ProductSet]

                self.Instance.YearlyStandardDevDemands = [1 if p in finishproduct else 0 for p in
                                                          self.Instance.ProductSet]

            if self.Instance.Distribution == Constants.Binomial:
                self.Instance.YearlyAverageDemand = [3.5 if p in finishproduct else 0 for p in self.Instance.ProductSet]

                self.Instance.YearlyStandardDevDemands = [1 if p in finishproduct else 0 for p in
                                                          self.Instance.ProductSet]

            if self.Instance.Distribution == Constants.Uniform:
                self.Instance.YearlyAverageDemand = [0.5 if p in finishproduct else 0 for p in self.Instance.ProductSet]

            self.Instance.ForecastedAverageDemand = [[self.Instance.YearlyAverageDemand[p]
                                                      if t >= self.Instance.NrTimeBucketWithoutUncertaintyBefore
                                                      else 0
                                                      for p in self.Instance.ProductSet]
                                                     for t in self.Instance.TimeBucketSet]

            self.Instance.ForcastedStandardDeviation = [[self.Instance.YearlyStandardDevDemands[p]
                                                         if t >= self.Instance.NrTimeBucketWithoutUncertaintyBefore
                                                         else 0
                                                         for p in self.Instance.ProductSet]
                                                        for t in self.Instance.TimeBucketSet]
            self.Instance.ForecastError = [-1 for t in self.Instance.TimeBucketSet]
            self.Instance.RateOfKnownDemand = 0.0

    def GetProductLevel(self):
        result = [ 0 for i in  self.Instance.ProductSet ]
        existdeeperproduct = True

        while existdeeperproduct :
            existdeeperproduct = False
            for p  in self.Instance.ProductSet:
                for q in self.Instance.ProductSet:
                    if self.Instance.Requirements[p][q] > 0 and  result[q] <= result[p]:
                        result[q] = result[p] + 1
                        existdeeperproduct = True

        return result
