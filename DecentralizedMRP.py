#This class provide methods to plan according to simple rules (EOQ, POQ, ...)
# It also include the computation of safety stock
# and a method to shift the production quantities in order to respect the capacity
from __future__ import absolute_import, division, print_function

from Constants import Constants
from ScenarioTreeNode import ScenarioTreeNode
from Solution import Solution
#from Solver import Solver
import math

#This object contains logic and methods to compute the classical MRP in decentralized fashion
class DecentralizedMRP(object):


    # constructor
    def __init__(self,  mrpinstance, safetystocksrave ):
        self.Instance =mrpinstance
        self.Solution = None
        self.SafetyStock = None
        self.EOQValues = None
        self.FixUntil = -1
        self.FixedSetup = False
        self.UseSSGraveInRules = safetystocksrave

        #This array indicates whether a produt and time period have already been planned
        self.Planned = [[False for p in self.Instance.ProductSet] for t in self.Instance.TimeBucketSet]
        #Compute preliminary values
        if self.UseSSGraveInRules:
            self.SafetyStock = self.ComputeSafetyStockGrave()

            #Remove safety stock at the end of horizon
            timetoenditem = self.Instance.GetTimeToEnd()
            # print "timetoenditem %s"%timetoenditem
            for p in self.Instance.ProductSet:
                for t in self.Instance.TimeBucketSet:
                    if self.Instance.MaximumQuanityatT[t][p] <= self.SafetyStock[t][p] \
                                or (self.Instance.NrTimeBucket - t <= timetoenditem[p] and self.Instance.ActualEndOfHorizon):
                        self.SafetyStock[t][p]=0
        else:
            self.SafetyStock = self.ComputeSafetyStock()


    def ComputeDependentDemandBasedOnProjectedInventory(self, product):
        result = [0 for t in self.Instance.TimeBucketSet]

        previousdemand = 0
        for t in self.Instance.TimeBucketSet:
                    projectedbackorder, projectedinventory = self.GetProjetedInventory(t)
                    #result[t] += -min(projectedinventory[product], 0) + previousprojected
                    #previousprojected = min(projectedinventory[product], 0)
                    if self.UseSSGraveInRules:
                        demand = max(( self.SafetyStock[t][product] -  projectedinventory[product] ), 0)
                    else:
                        demand = max((- projectedinventory[product] ), 0)
                    result[t] +=  demand - previousdemand
                    previousdemand = demand



        if self.FixUntil + 2 + self.Instance.Leadtimes[product] < self.Instance.NrTimeBucket:
            result[self.FixUntil + 1 + self.Instance.Leadtimes[product]] = sum(
                result[tau] for tau in range(self.FixUntil + 1, self.FixUntil + 2 + self.Instance.Leadtimes[product]))


        #Do not consider negative demand
        for t in self.Instance.TimeBucketSet:
            result[t] = max( result[t], 0.0)

        return result


    def ComputeDependentDemand( self, product ):
        demand = [ self.Solution.Scenarioset[0].Demands[t][product] for t in self.Instance.TimeBucketSet ]

        levelset = sorted(set(self.Instance.Level), reverse=False)

        for l in levelset:
            prodinlevel = [p for p in self.Instance.ProductSet if self.Instance.Level[p] == l]
            for p in prodinlevel:
                for t in self.Instance.TimeBucketSet:
                    demand[t] += self.Solution.ProductionQuantity[0][t][p] * self.Instance.Requirements[p][product]

        return demand


    def GetServiceLevel(self, p):
        return  float(self.Instance.BackorderCosts[p] ) / float((self.Instance.BackorderCosts[p] + self.Instance.InventoryCosts[p] ) )

    def GetServiceLevelFromLostSale(self, p):
        return  float(self.Instance.LostSaleCost[p]) / float(
            (self.Instance.LostSaleCost[p] + self.Instance.InventoryCosts[p]))

    def GetMaxDemanWithRespectToServiceLevel(self, p, t, WithLosale=False):

        if not self.Instance.HasExternalDemand[p]:
            result = sum(self.GetMaxDemanWithRespectToServiceLevel(q, t) * self.Instance.Requirements[q][p]
                            for q in self.Instance.ProductSet if self.Instance.Requirements[q][p] > 0 )
        else:
            if WithLosale and t >= self.Instance.NrTimeBucket - 1:
                ratio = self.GetServiceLevelFromLostSale(p)
            else :
                ratio = self.GetServiceLevel(p)
            result = ScenarioTreeNode.TransformInverse(   [[ratio]],
                                                          1,
                                                          1,
                                                          self.Instance.Distribution,
                                                          [self.Instance.ForecastedAverageDemand[t][p]],
                                                          [self.Instance.ForcastedStandardDeviation[t][p]])[0][0]

        return result

    def ComputeSafetyStock(self):

        safetystock = [ [ 0.0 for p in self.Instance.ProductSet] for t in self.Instance.TimeBucketSet ]
        for p in self.Instance.ProductWithExternalDemand:
            for t in range(self.FixUntil+1, self.Instance.NrTimeBucket):
                safetystock[t][p] = self.GetMaxDemanWithRespectToServiceLevel(p, t) - self.Instance.ForecastedAverageDemand[t][p]

      #  print "safetystock %s" % safetystock
        return safetystock

    def GetDependentAverageDemand(self, p, t):
        if not self.Instance.HasExternalDemand[p]:
            result = sum(self.GetDependentAverageDemand(q, t) * self.Instance.Requirements[q][p]
                            for q in self.Instance.ProductSet if self.Instance.Requirements[q][p] > 0 )
        else:
            result = self.Instance.ForecastedAverageDemand[t][p]

        return result


    #This function return the quantity to remove to make the plan feasible according to capacities
    def CheckCapacity( self, p, t ):
        result = -Constants.Infinity
        #for each resource:
        for k in self.Instance.ResourceSet:
            if self.Instance.ProcessingTime[p][k] > 0:
                #compute the capacity violation
                capacityconsumption = sum( self.Instance.ProcessingTime[q][k] * self.Solution.ProductionQuantity[0][t][q] for q in self.Instance.ProductSet)
                violation = capacityconsumption - self.Instance.Capacity[k]
                #Compute the quantity violating
                quantityviolation = violation / self.Instance.ProcessingTime[p][k]


                #record the largest violating quantity
                if result < quantityviolation:
                    result = quantityviolation

        return result

     # This function return the quantity to remove to make the plan feasible according to requirementt in components
    def CheckRequirement(self, p, t):
            result = -Constants.Infinity
            # for each resource:
            for q in self.Instance.ProductSet:
                if self.Instance.Requirements[p][q] > 0 and (t-self.Instance.Leadtimes[q] < 0 or self.Planned[t-self.Instance.Leadtimes[q]][q]):
                    #Compute the quantity of q reuire to produce p
                    requiredquantity = self.Instance.Requirements[p][q] * self.Solution.ProductionQuantity[0][t][p]

                   # if (t - self.Instance.Leadtimes[q]) >= 0 :
                    projectedbackorder, projectedinventory = self.GetProjetedInventory( t )

                    quantityviolation = min(requiredquantity, -( projectedinventory[q] / self.Instance.Requirements[p][q] ) )
                    #else:
                    #    quantityviolation = requiredquantity
                    if result < quantityviolation:
                        result = quantityviolation

            return result

    def CheckFixedSetup(self, p, t):
        if self.FixedSetup and self.Solution.Production[0][t][p] == 0:
                return 0
        else:
            return Constants.Infinity

    def GetViolation(self, p, t):
        result = max( self.CheckRequirement( p, t ), self.CheckCapacity( p, t ) )
        return result

    # This function return the quantity to remove to make the plan feasible according to capacities
    def MoveBackward(self, quantity,  p, t):

        bestspreading = [0 for tau in range(t) ]
        bestspreadingcost = Constants.Infinity

        #For each time period compute the cost of spreading teh quantity from that point
        for tau in range(t):
            #while the remaining quantity is positive, replan as much as possible in the earliest period
            remainingquantity = quantity
            challengingspreading = [0 for tau2 in range(t) ]
            for tau2 in reversed(range(  self.FixUntil+1, tau + 1)):
                quantityreplanattau2 = min( -self.CheckCapacity(p, tau2), remainingquantity, self.CheckFixedSetup(p, tau2))
                challengingspreading[tau2] = quantityreplanattau2
                remainingquantity = remainingquantity - quantityreplanattau2

            if remainingquantity == 0:
                cost = self.ComputeCostReplan(challengingspreading, p, t)
                if cost < bestspreadingcost:
                    bestspreading = [ challengingspreading[tau2]  for tau2 in range(t) ]
                    bestspreadingcost = cost
            else:
                if Constants.Debug:
                    print("spreading non feasible")

        #No feasible solution were found
        if bestspreadingcost == Constants.Infinity:
            return False

        else:
            if Constants.Debug:
                print("chosen spreading %r" % bestspreading)
            self.Solution.ProductionQuantity[0][t][p] -= quantity
            for tau in range(t):
                self.Solution.ProductionQuantity[0][tau][p] +=  bestspreading[tau]
            return True

    def ComputeCostReplan( self, spreading, p, t ):
        cost = 0
        #for each period add the inventory cost and setup
        for tau in range(t):
            inventorycost = spreading[tau] * ( (t+1) - tau ) * self.Instance.InventoryCosts[p]
            if spreading[tau] > 0 and self.Solution.ProductionQuantity[0][tau][p] == 0:
                setupcost = self.Instance.SetupCosts[p]
            else: setupcost = 0
            cost = cost + inventorycost + setupcost
        return cost


    # This function return the quantity to remove to make the plan feasible according to capacities
    def MoveForward(self, quantity, prod, time):
        remainingquantity = quantity
        self.Solution.ProductionQuantity[0][time][prod] -= remainingquantity
        #for each time period greater than t, move as much as possible
        for tau in range(time, self.Instance.NrTimeBucket):
            #get the quantity to move if the
            quantityreplanattau2 = min(-self.CheckCapacity(prod, tau), remainingquantity)
            self.Solution.ProductionQuantity[0][tau][prod] += quantityreplanattau2
            remainingquantity = remainingquantity - quantityreplanattau2


        self.RepairRequirement( prod )

    #This functioninfer the value of Y from the value of Q
    def InferY( self ):
        self.Solution.Production = [ [ [ 1 if self.Solution.ProductionQuantity[0][t][p] > 0 else 0
                                         for p in self.Instance.ProductSet]
                                         for t in self.Instance.TimeBucketSet ] ]


    #This functioninfer the value of Y from the value of Q
    def InferInventory( self ):
        self.Solution.InventoryLevel = [ [ [ 0 for p in self.Instance.ProductSet ]
                                           for t in self.Instance.TimeBucketSet ] ]

        self.Solution.BackOrder = [[ [ 0 for p in self.Instance.ProductWithExternalDemand ]
                                     for t in self.Instance.TimeBucketSet ] ]
        for t in self.Instance.TimeBucketSet:
            backorder, inventory = self.GetProjetedInventory( t )
            for p in self.Instance.ProductSet:
                self.Solution.InventoryLevel[0][t][p] = max( inventory[p], 0 )
                if self.Instance.HasExternalDemand[p]:
                    self.Solution.BackOrder[0][t][ self.Instance.ProductWithExternalDemandIndex[p] ] =  max( -inventory[ p ], 0 )