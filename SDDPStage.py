from __future__ import absolute_import, division, print_function
import cplex
import math
from Constants import Constants
from SDDPCut import SDDPCut
from ScenarioTree import ScenarioTree
import itertools

# This class contains the attributes and methodss allowing to define one stage of the SDDP algorithm.
class SDDPStage(object):

    def __init__(self, owner=None, previousstage=None, nextstage=None, decisionstage=-1, fixedccenarioset=[],
                 forwardstage = None, isforward = False ):

        self.SDDPOwner = owner
        self.PreviousSDDPStage = previousstage
        self.NextSDDPStage = nextstage
        self.CorrespondingForwardStage = forwardstage
        self.IsForward = isforward
        self.SDDPCuts = []
        #A Cplex MIP object will be associated to each stage later.
        self.Cplex = cplex.Cplex()
        #The variable MIPDefined is turned to True when the MIP is built
        self.MIPDefined = False
        self.DecisionStage = decisionstage
        #self.TimeDecisionStage corresponds to the period of the first quantity variables in the stage
        self.TimeDecisionStage = -1
        self.Instance = self.SDDPOwner.Instance

        #The following attribute will contain the coefficient of hte variable in the cuts
        self.CoefficientConstraint = []
        #The following table constains the value at which the variables are fixed
        self.VariableFixedTo = []


        self.EVPIScenarioSet = None
        self.EVPIScenarioRange = range(Constants.SDDPNrEVPIScenario)

        # Set of scenario used to build the MIP
        self.FixedScenarioSet = fixedccenarioset

        self.FixedScenarioPobability = []

        #The number of variable of each type in the stage will be set later
        self.NrProductionVariable = 0
        self.NrQuantityVariable = 0
        self.NrConsumptionVariable = 0
        self.NrStockVariable = 0
        self.NrBackOrderVariable = 0
        self.NrCostToGo = 0
        self.NRFlowFromPreviousStage = 0
        self.NRProductionRHS = 0
        self.NrEstmateCostToGoPerItemPeriod = 0
        self.NrPIQuantity = 0
        self.NrPIInventory = 0
        self.NrPIBacklog = 0
        self.NrPIConsumption = 0
        self.NrEstmateCostToGoEVPI = 0
        self.NrPIFlowFromPreviouQty = 0
        # The start of the index of each variable
        self.StartProduction = 0
        self.StartQuantity = 0
        self.StartConsumption = 0
        self.StartStock = 0
        self.StartBackOrder = 0
        self.StartCostToGo = 0
        self.StartFlowFromPreviousStage = 0
        self.StartProductionRHS = 0
        self.StartEstmateCostToGoPerItemPeriod = 0
        self.StartPIQuantity = 0
        self.StartPIInventory = 0
        self.StartPIBacklog = 0
        self.StartPIConsumption = 0
        self.StartPIFlowFromPreviouQty = 0
        self.StartPICostToGoEVPI = 0
        self.StartCutRHSVariable = 0
        #self.StartZ = self.StartEstmateCostToGoPerItemPeriod + self.NrEstmateCostToGoPerItemPeriod
        # Demand and materials requirement: set the value of the invetory level and backorder quantity according to
        #  the quantities produced and the demand
        self.M = cplex.infinity
        self.CurrentTrialNr = -1
        #The quantity to order (filled after having solve the MIPs for all scenario)
        self.QuantityValues = []
        #The value of the production variables (filled after having solve the MIPs for all scenario)
        self.ProductionValue = []
        # The value of the inventory variables (filled after having solve the MIPs for all scenario)
        self.InventoryValue = []
        #The value of the backorder variable (filled after having solve the MIPs for all scenario)
        self.BackorderValue = []
        # The consumption  (filled after having solve the MIPs for all scenario)
        self.ConsumptionValues = []

        # Try to use the corepoint method of papadakos, remove if it doesn't work
        self.CorePointQuantityValues = []
        # The value of the production variables (filled after having solve the MIPs for all scenario)
        self.CorePointProductionValue = []
        # The value of the inventory variables (filled after having solve the MIPs for all scenario)
        self.CorePointInventoryValue = []
        # The value of the backorder variable (filled after having solve the MIPs for all scenario)
        self.CorePointBackorderValue = []

        #The cost of each scenario
        self.StageCostPerScenarioWithoutCostoGo = []
        self.StageCostPerScenarioWithCostoGo = []
        self.PartialCostPerScenario = []
        self.PassCost = -1
        self.NrTrialScenario = -1

        self.LastAddedConstraintIndex = 0
        self.IndexFlowConstraint = []
        self.IndexProductionQuantityConstraint = []
        self.IndexCapacityConstraint = []
        self.IndexCutConstraint = []
        self.IndexConsumptionConstraint = []
        self.IndexPIFlowConstraint = []
        self.IndexPIProductionQuantityConstraint = []
        self.IndexPICapacityConstraint = []
        self.IndexPIConsumptionConstraint = []
        self.ConcernedProductFlowConstraint = []
        self.ConcernedScenarioFlowConstraint = []
        self.ConcernedTimeFlowConstraint = []
        self.ConcernedScenarioProductionQuantityConstraint = []
        self.ConcernedProductProductionQuantityConstraint = []
        self.ConcernedTimeProductionQuantityConstraint = []
        self.ConcernedResourceCapacityConstraint = []
        self.ConcernedScenarioCapacityConstraint = []

        self.ConcernedProductPIFlowConstraint = []
        self.ConcernedScenarioPIFlowConstraint = []
        self.ConcernedEVPIScenarioPIFlowConstraint = []
        self.ConcernedTimePIFlowConstraint = []
        self.ConcernedScenarioPIProductionQuantityConstraint = []
        self.ConcernedProductPIProductionQuantityConstraint = []
        self.ConcernedTimePIProductionQuantityConstraint = []
        self.ConcernedResourcePICapacityConstraint = []
        self.ConcernedScenarioPICapacityConstraint = []


        self.IndexCutConstraint = []
        self.ConcernedCutinConstraint = []
        self.RangePeriodQty = []
        self.RangePeriodEndItemInv = []
        self.RangePeriodComponentInv = []
        self.RangePeriodInv = []
        self.PeriodsInGlobalMIPQty = []
        self.PeriodsInGlobalMIPEndItemInv = []
        self.PeriodsInGlobalMIPComponentInv = []
        self.PeriodsInGlobalMIPInv = []

        self.StockIndexArray = []

    def ComputeVariableIndices(self):
        self.TimePeriodToGoQty = range(self.TimeDecisionStage + len(self.RangePeriodQty), self.Instance.NrTimeBucket)
        self.TimePeriodToGoEndItInv = range(self.TimeDecisionStage + len(self.RangePeriodQty) -1, self.Instance.NrTimeBucket)
        self.TimePeriodToGoCompInv = range(self.TimeDecisionStage + len(self.RangePeriodQty), self.Instance.NrTimeBucket)

        self.TimePeriodToGo = range(self.TimeDecisionStage, self.Instance.NrTimeBucket)
        self.NrTimePeriodToGo = len(self.TimePeriodToGo)

        self.ComputeNrVariables()
        # The start of the index of each variable
        self.StartProduction = 0
        self.StartQuantity = self.StartProduction + self.NrProductionVariable
        self.StartConsumption = self.StartQuantity + self.NrQuantityVariable
        self.StartStock = self.StartConsumption + self.NrConsumptionVariable
        self.StartBackOrder = self.StartStock + self.NrStockVariable
        self.StartCostToGo = self.StartBackOrder + self.NrBackOrderVariable
        self.StartFlowFromPreviousStage = self.StartCostToGo + self.NrCostToGo
        self.StartProductionRHS = self.StartFlowFromPreviousStage + self.NRFlowFromPreviousStage
        self.StartEstmateCostToGoPerItemPeriod = self.StartProductionRHS + self.NRProductionRHS
        self.StartPIQuantity = self.StartEstmateCostToGoPerItemPeriod + self.NrEstmateCostToGoPerItemPeriod
        self.StartPIInventory = self.StartPIQuantity + self.NrPIQuantity
        self.StartPIBacklog = self.StartPIInventory + self.NrPIInventory
        self.StartPIConsumption = self.StartPIBacklog + self.NrPIBacklog
        self.StartPIFlowFromPreviouQty = self.StartPIConsumption + self.NrPIConsumption
        self.StartPICostToGoEVPI = self.StartPIFlowFromPreviouQty + self.NrPIFlowFromPreviouQty
        self.StartCutRHSVariable = self.StartPICostToGoEVPI + self.NrEstmateCostToGoEVPI

    #Compute the cost of the stage at the current iteration
    def ComputePassCost(self):
        self.PassCost = sum(self.PartialCostPerScenario[w] for w in self.TrialScenarioNrSet) \
                        / self.NrTrialScenario

        self.PassCostWithAproxCosttoGo = sum(self.StageCostPerScenarioWithCostoGo[w]
                                             for w in self.TrialScenarioNrSet) / self.NrTrialScenario

    #This function modify the number of scenario in the stage
    def SetNrTrialScenario(self, nrscenario):
        self.NrTrialScenario = nrscenario
        self.CurrentTrialNr = 0
        self.TrialScenarioNrSet = range(nrscenario)
        #The quantity to order (filled after having solve the MIPs for all scenario)
        self.QuantityValues = [[] for w in self.TrialScenarioNrSet]
        # The quantity to order (filled after having solve the MIPs for all scenario)
        self.ConsumptionValues = [[] for w in self.TrialScenarioNrSet]
        #The value of the production variables (filled after having solve the MIPs for all scenario)
        self.ProductionValue = [[[] for t in self.Instance.TimeBucketSet] for w in self.TrialScenarioNrSet]
        # The value of the inventory variables (filled after having solve the MIPs for all scenario)
        self.InventoryValue = [[] for w in self.TrialScenarioNrSet]
        #The value of the backorder variable (filled after having solve the MIPs for all scenario)
        self.BackorderValue = [[] for w in self.TrialScenarioNrSet]
        # The cost of each scenario
        self.StageCostPerScenarioWithoutCostoGo = [-1 for w in self.TrialScenarioNrSet]
        self.StageCostPerScenarioWithCostoGo = [-1 for w in self.TrialScenarioNrSet]

        self.PartialCostPerScenario = [0 for w in self.TrialScenarioNrSet]


    #Return true if the current stage is the last
    def IsLastStage(self):
        return False

    #Return true if the current stage is the first
    def IsFirstStage(self):
        return self.DecisionStage == 0

        # Return true if the current stage is the first

    def IsPenultimateStage(self):
        return False#self.NextSDDPStage.IsLastStage()

    def IsInFutureStageForInv(self, p, t):
        return  not ( (self.Instance.HasExternalDemand[p] and t in self.PeriodsInGlobalMIPEndItemInv)
                or ((not self.Instance.HasExternalDemand[p]) and t in self.PeriodsInGlobalMIPComponentInv))

    def GetNumberOfPeriodWithQuantity(self):
        if self.IsFirstStage():
            return range(self.Instance.NrTimeBucketWithoutUncertaintyBefore + 1)

        if self.IsPenultimateStage():
            return range(self.Instance.NrTimeBucket - self.DecisionStage - self.Instance.NrTimeBucketWithoutUncertaintyBefore -1)
        elif not self.IsLastStage():
            return [0]
        else:
            return []

    def GetNumberOfPeriodWithEndItemInventory(self):
        if self.IsFirstStage():
            return range(1, self.Instance.NrTimeBucketWithoutUncertaintyBefore+1)
        elif not self.IsLastStage():
            return [0]
        else:
            return range(self.Instance.NrTimeBucket - self.DecisionStage - self.Instance.NrTimeBucketWithoutUncertaintyBefore + 1)

    def GetNumberOfPeriodWithComponentInventory(self):
        if self.IsFirstStage():
            return range(self.Instance.NrTimeBucketWithoutUncertaintyBefore +1)
        elif not self.IsLastStage():
            return [0]
        else:
            return range(
                self.Instance.NrTimeBucket - self.DecisionStage - self.Instance.NrTimeBucketWithoutUncertaintyBefore -1)

    #Compute the number of variable of each type (production quantty, setup, inventory, backorder)
    def ComputeNrVariables(self):
        #number of variable at stage 1<t<T
        NrFixedScen = len(self.FixedScenarioSet)
        NrPeriodQty = len(self.RangePeriodQty)
        NrEndItInvPeriod = len(self.RangePeriodEndItemInv)
        NrCompInvPeriod = len(self.RangePeriodComponentInv)
        self.NrBackOrderVariable = len(self.Instance.ProductWithExternalDemand) * NrFixedScen * NrEndItInvPeriod
        self.NrQuantityVariable = self.Instance.NrProduct * NrFixedScen * NrPeriodQty
        self.NrConsumptionVariable = len(self.Instance.ConsumptionSet) * NrFixedScen * NrPeriodQty
        self.NrStockVariable = len(self.Instance.ProductWithExternalDemand) * NrFixedScen * NrEndItInvPeriod \
                               + len(self.Instance.ProductWithoutExternalDemand) * NrFixedScen * NrCompInvPeriod \

        self.NrProductionVariable = 0
        self.NrEstmateCostToGoPerItemPeriod = 0

        self.NrCostToGo = NrFixedScen
        if self.IsLastStage():
            self.NrCostToGo = 0
        self.NRFlowFromPreviousStage = int(self.NrStockVariable/NrFixedScen)
        self.NRProductionRHS = self.Instance.NrProduct * NrPeriodQty
        nrtimeperiodtogo = self.Instance.NrTimeBucket - (self.TimeDecisionStage )
        nrtimeperiodtogoqty = (nrtimeperiodtogo - len(self.RangePeriodQty))
        if Constants.SDDPUseEVPI and not self.IsLastStage():
            self.NrPIQuantity = NrFixedScen * Constants.SDDPNrEVPIScenario * nrtimeperiodtogoqty * self.Instance.NrProduct
            self.NrPIInventory = NrFixedScen * Constants.SDDPNrEVPIScenario * (nrtimeperiodtogoqty + 1 ) * len(self.Instance.ProductWithExternalDemand) \
                                     + NrFixedScen * Constants.SDDPNrEVPIScenario * (nrtimeperiodtogoqty) * len(self.Instance.ProductWithoutExternalDemand)

            self.NrPIBacklog = NrFixedScen * Constants.SDDPNrEVPIScenario * (nrtimeperiodtogoqty + 1) * len(self.Instance.ProductWithExternalDemand)
            self.NrPIConsumption = NrFixedScen * Constants.SDDPNrEVPIScenario * (nrtimeperiodtogoqty) * len(self.Instance.ConsumptionSet)
            self.NrEstmateCostToGoEVPI = NrFixedScen
            self.NRProductionRHS = self.Instance.NrProduct * nrtimeperiodtogo
            self.NrPIFlowFromPreviouQty = self.Instance.NrProduct * (self.Instance.NrTimeBucket - (nrtimeperiodtogo))

        # self.NrZ = 0
        # number of variable at stage 1


        if self.IsFirstStage():
            self.NrProductionVariable = self.Instance.NrTimeBucket * self.Instance.NrProduct
           # self.NrBackOrderVariable = NrInvPeriod * len(self.Instance.ProductWithExternalDemand)
            self.NRProductionRHS = 0
           # self.NrStockVariable = (NrInvPeriod + 1) * len(self.Instance.ProductWithoutExternalDemand) \
           #                        + (NrInvPeriod -1) * len(self.Instance.ProductWithExternalDemand)
           # self.NRFlowFromPreviousStage = self.NrStockVariable

            #if Constants.SDDPUseValidInequalities:
            #    self.NrEstmateCostToGoPerItemPeriod = self.Instance.NrTimeBucket * len(self.Instance.ProductWithExternalDemand)
               # self.NrZ = self.Instance.NrTimeBucket * self.Instance.NrTimeBucket * self.Instance.NrProduct


    # return the index of the cost to go variable for scenario w
    def GetIndexCostToGo(self, w):
            return self.StartCostToGo + w

    def GetIndexEVPICostToGo(self, w):
        return self.StartPICostToGoEVPI + w

    #return the index of the production variable associated with the product p at time t
    def GetIndexProductionVariable(self, p, t):
        if self.IsFirstStage():
            return self.StartProduction + p*self.Instance.NrTimeBucket + t
        else:
            raise ValueError('Production variables are only defined at stage 0')

    # return the index of the production variable associated with the product p at time t
    def GetIndexFlowFromPreviousStage(self, p, t):
        return self.FlowIndexArray[t][p]

            # return the index of the production variable associated with the product p at time t

    def GetIndexPIFlowPrevQty(self, p, t):
        return self.StartPIFlowFromPreviouQty + t*self.Instance.NrProduct + p

    def GetIndexProductionRHS(self, p, t):
        return self.StartProductionRHS + t * self.Instance.NrProduct + p

    def GetIndexPIProductionRHS(self, p, t):
        if t <= self.TimeDecisionStage + len(self.RangePeriodQty):
            return self.GetIndexProductionRHS(p, t-self.TimeDecisionStage )
        else:
            return self.StartProductionRHS  \
                    + (t - (self.TimeDecisionStage)) * self.Instance.NrProduct \
                    + p

    #In this function t is the global time index not the one spcific to the decision stage
    def GetIndexPIQuantityVariable(self, p, t,  wevpi, w):
        if t in self.PeriodsInGlobalMIPQty:
            time = self.GetTimeIndexForQty(p, t)
            return self.GetIndexQuantityVariable(p, time, w)
        else:
            timeperiodwithqty = self.NrTimePeriodToGo - len(self.RangePeriodQty)
            return self.StartPIQuantity \
                    + w * self.Instance.NrProduct * timeperiodwithqty * Constants.SDDPNrEVPIScenario \
                    + p * timeperiodwithqty * Constants.SDDPNrEVPIScenario \
                    + wevpi * timeperiodwithqty \
                    + (t-(self.TimeDecisionStage+ len(self.RangePeriodQty)))

    def GetIndexPIInventoryVariable(self, p, t,  wevpi, w):
        if not self.IsInFutureStageForInv(p, t):
            time = self.GetTimeIndexForInv(p, t)
            return self.GetIndexStockVariable(p, time, w)
        else:
            nrperscenar = len(self.Instance.ProductWithExternalDemand) * len(self.TimePeriodToGoEndItInv) \
            + len(self.Instance.ProductWithoutExternalDemand) * len(self.TimePeriodToGoCompInv)
            reshapetime = self.TimePeriodToGoEndItInv[0] #len(self.RangePeriodEndItemInv)
            if not self.Instance.HasExternalDemand[p]:
                reshapetime = self.TimePeriodToGoCompInv[0]

            return self.StartPIInventory \
                   + w * nrperscenar * Constants.SDDPNrEVPIScenario \
                   + wevpi * nrperscenar \
                   + sum((len(self.TimePeriodToGoEndItInv))
                         if self.Instance.HasExternalDemand[q]
                         else (len(self.TimePeriodToGoCompInv))
                         for q in range(0, p)) \
                   + t-( reshapetime)


    def GetIndexPIBacklogVariable(self, p, t,  wevpi, w):
        if t in self.PeriodsInGlobalMIPEndItemInv:
            time = self.GetTimeIndexForBackorder(p, t)
            return self.GetIndexBackorderVariable(p, time, w)
        else:
            nrtimeperiodwithbackorder = len(self.TimePeriodToGoEndItInv)
            return self.StartPIBacklog + \
                       + w * len(self.Instance.ProductWithExternalDemand) * nrtimeperiodwithbackorder * Constants.SDDPNrEVPIScenario \
                       + self.Instance.ProductWithExternalDemandIndex[p] * nrtimeperiodwithbackorder * Constants.SDDPNrEVPIScenario \
                       + wevpi * (self.NrTimePeriodToGo) \
                       + t-(self.TimePeriodToGoEndItInv[0])

    def GetIndexPIConsumptionVariable(self, c, t, wevpi, w):
        if t in self.PeriodsInGlobalMIPQty:
            time = self.GetTimeIndexForQty(c[0], t)
            return self.GetIndexConsumptionVariable(c, time, w)
        else:
            timeperiodwithqty = self.NrTimePeriodToGo - len(self.RangePeriodQty)
            return self.StartPIConsumption + \
                   + w * len(self.Instance.ConsumptionSet) * timeperiodwithqty * Constants.SDDPNrEVPIScenario \
                   + self.Instance.ConsumptionSet.index(c) * timeperiodwithqty * Constants.SDDPNrEVPIScenario  \
                   + wevpi * (timeperiodwithqty) \
                   + t-(self.TimeDecisionStage+len(self.RangePeriodQty))

    #return the index of the production variable associated with the product p at time t
    #def GetIndexZVariable(self, p, t, t2):
    #    if self.IsFirstStage():
    #        return self.StartZ + p*self.Instance.NrTimeBucket * self.Instance.NrTimeBucket+ t * self.Instance.NrTimeBucket +t2
    #    else:
    #        raise ValueError('Production variables are only defined at stage 0')

    def GetIndexEstmateCostToGoPerItemPeriod(self, p, t):
        if self.IsFirstStage() and Constants.SDDPUseValidInequalities:
            return self.StartEstmateCostToGoPerItemPeriod + self.Instance.ProductWithExternalDemandIndex[p] * self.Instance.NrTimeBucket + t
        else:
            raise ValueError('EstmateCostToGoPerItemPeriod variables are only defined at stage 0 with valid inequalitties')


    #Return the index of the variable associated with the quanity of product p decided at the current stage
    def GetIndexQuantityVariable(self, p, t, w):
        return self.StartQuantity \
               + t * len(self.FixedScenarioSet) *self.Instance.NrProduct \
               + w * self.Instance.NrProduct \
               + p

    #Return the index of the variable associated with the quanity of product p decided at the current stage
    def GetIndexConsumptionVariable(self, c, t, w):
        return self.StartConsumption \
               + t * len(self.FixedScenarioSet) * len(self.Instance.ConsumptionSet) \
               + w * len(self.Instance.ConsumptionSet)\
               + self.Instance.ConsumptionSet.index(c)

    def ComputeFlowIndexArray(self):
        i = self.StartFlowFromPreviousStage
        self.FlowIndexArray = [[-1 for p in self.Instance.ProductSet]
                                 for t in self.RangePeriodInv]

        for t in self.RangePeriodInv:
                for p in self.GetProductWithStockVariable(t):
                    self.FlowIndexArray[t][p] = i
                    i = i + 1


    def ComputeStockIndexArray(self):
        i = self.StartStock
        self.StockIndexArray = [[[-1 for p in self.Instance.ProductSet]
                                 for w in self.FixedScenarioSet]
                                for t in self.RangePeriodInv]

        for t in self.RangePeriodInv:
            for w in self.FixedScenarioSet:
                for p in self.GetProductWithStockVariable(t):
                    self.StockIndexArray[t][w][p] = i
                    i = i + 1

    #Return the index of the variable associated with the stock of product p decided at the current stage
    def GetIndexStockVariable(self, p, t, w):
        return self.StockIndexArray[t][w][p]

    def GetNamePIStockVariable(self, p, t, wevpi, w):
            return "PII_%d_%d_%d_%d" % (p, t, wevpi, w)

    def GetNamePIConsumptionVariable(self, c, t, wevpi, w):
            return "PIW_%da%dt%ds%d_%d" % (c[0], c[1], t, wevpi, w)

    def GetNamePIQuantityVariable(self, p, t, wevpi, w):
            return "PIQ_%d_%d_%d_%d" % (p, t, wevpi, w)

    def GetNamePIBacklogVariable(self, p, t, wevpi, w):
        return "PIB_%d_%d_%d_%d" % (p, t, wevpi, w)

    def GetNamePIFlowPrevQty(self, p, t):
        return "PIprevQty_%d_%d"%(p, t)

    #Return the index of the variable associated with the stock of product p decided at the current stage
    def GetIndexBackorderVariable(self, p, t,  w):
        if self.IsFirstStage() and t == 0:
            raise ValueError('Backorder variables are not defined at stage 0')
        else:
            shift = 0
            if self.IsFirstStage():
                shift = 1

            return self.StartBackOrder + \
                   + (t - shift) * len(self.FixedScenarioSet) * len(self.Instance.ProductWithExternalDemand) \
                   + w * len(self.Instance.ProductWithExternalDemand) \
                   + self.Instance.ProductWithExternalDemandIndex[p]

    def GetIndexCutRHSFromPreviousSatge(self, cut):
        return self.StartCutRHSVariable + cut.Id

    #Return the name of the variable associated with the setup of product p at time t
    def GetNameProductionVariable(self, p, t):
        if self.IsFirstStage():
            return "Y_%d_%d"%(p, t)
        else:
            raise ValueError('Production variables are only defined at stage 0')

    def GetNameEstmateCostToGoPerItemPeriod(self, p, t):
        if self.IsFirstStage() and Constants.SDDPUseValidInequalities:
            return "e_%d_%d"%(p, t)
        else:
            raise ValueError('EstmateCostToGoPerItemPeriod variables are only defined at stage 0 with valid inequalitties')

    # Return the name of the variable associated with the quanity of product p decided at the current stage
    def GetNameQuantityVariable(self, p, t, w):
        time = self.GetTimePeriodAssociatedToQuantityVariable(p, t)
        return "Q_%d_%d_%d"%(p, time, w)

    def GetNameFlowFromPreviousStage(self, p, t):
        time = self.GetTimePeriodAssociatedToInventoryVariable(p, t)
        return "Flow_%r_%r" % (p, time)

    def GetNameProductionRHS(self, p, t):
        return "prodRHS_%r_%r" % (p,t)

    # Return the name of the variable associated with the consumption c at the current stage
    def GetNameConsumptionVariable(self, c, t, w):
        time = self.GetTimePeriodAssociatedToQuantityVariable(c[1], t)
        return "W_%dto%d_%d_%d"%(c[0], c[1], time, w)

    # Return the name of the variable associated with the stock of product p decided at the current stage
    def GetNameStockVariable(self, p, t, w):
        time = self.GetTimePeriodAssociatedToInventoryVariable(p, t)
        return "I_%d_%d_%d"%(p, time, w)

    # Return the name of the variable associated with the backorder of product p decided at the current stage
    def GetNameBackorderVariable(self, p, t, w):
         time = self.GetTimePeriodAssociatedToBackorderVariable(p, t)
         return "B_%d_%d_%d" % (p, time, w)


    def GetTimeIndexForQty(self, p , time):
        result = time - self.TimeDecisionStage
        return result

    # Return the time period associated with quanity of product p decided at the current stage
    def GetTimePeriodAssociatedToQuantityVariable(self, p, t):
        result = self.TimeDecisionStage + t
        return result

    def GetTimeIndexForInv(self, p, time):
        result = time - self.TimeDecisionStage + 1
        if not self.Instance.HasExternalDemand[p]:
            result = time - self.TimeDecisionStage
        return result


    # Return the time period associated with inventory of product p decided at the current stage (the inventory level of component,  at time t is observed at stage t -1 because it is not stochastic)
    def GetTimePeriodAssociatedToInventoryVariable(self, p, t):
        result = self.TimeDecisionStage + t - 1
        if not self.Instance.HasExternalDemand[p]:
            result = self.TimeDecisionStage + t

        return result

    def GetTimePeriodRangeForInventoryVariable(self, p):
        if self.Instance.HasExternalDemand[p]:
            result = [self.TimeDecisionStage + t - 1 for t in self.RangePeriodEndItemInv]
        else:
            result = [self.TimeDecisionStage + t for t in self.RangePeriodComponentInv]

        return result

    def GetTimeIndexForBackorder(self,p , time):
        result = time - self.TimeDecisionStage + 1
        return result

    # Return the time period associated with backorder of product p decided at the current stage
    def GetTimePeriodAssociatedToBackorderVariable(self, p, t):
        result = self.TimeDecisionStage + t - 1
        if not self.Instance.HasExternalDemand[p]:
            raise ValueError('Backorder variables are not defined for component')
        return result

    #This function return the right hand side of flow constraint for product p
    def GetRHSFlowConst(self, p, scenario, forwardpass):
        print("Should Not Enter HERE")
        righthandside = 0
        if self.Instance.HasExternalDemand[p] and not self.IsFirstStage():
            #Demand at period t for item i
            if forwardpass:
                righthandside = righthandside \
                               + self.SDDPOwner.CurrentSetOfTrialScenarios[scenario].Demands[
                                   self.GetTimePeriodAssociatedToInventoryVariable(p)][p]
            else:
                righthandside = righthandside \
                                + self.SDDPOwner.SetOfSAAScenario[self.GetTimePeriodAssociatedToInventoryVariable(p)][scenario][p]

            if self.GetTimePeriodAssociatedToBackorderVariable(p) -1 >= 0:
                #Backorder at period t-1
                righthandside = righthandside \
                                   + self.SDDPOwner.GetBackorderFixedEarlier(p,
                                                                             self.GetTimePeriodAssociatedToBackorderVariable(p) - 1,
                                                                             self.CurrentTrialNr)

        productionstartedtime = self.GetTimePeriodAssociatedToInventoryVariable(p) - self.Instance.LeadTimes[p]
        if productionstartedtime >= 0:
            #Production at period t-L_i
            righthandside = righthandside \
                               - self.SDDPOwner.GetQuantityFixedEarlier(p, productionstartedtime,
                                                                        self.CurrentTrialNr)

        if self.GetTimePeriodAssociatedToInventoryVariable(p) - 1 >= -1 and not (self.IsFirstStage() and self.Instance.HasExternalDemand[p] ):
            #inventory at period t- 1
            righthandside = righthandside \
                               - self.SDDPOwner.GetInventoryFixedEarlier(p,
                                                                         self.GetTimePeriodAssociatedToInventoryVariable(
                                                                             p) - 1, self.CurrentTrialNr)
        return righthandside

    def GetRHSFlow(self, p, t, scenario, forwardpass):

        righthandside = 0
        if self.Instance.HasExternalDemand[p]:# and not self.IsFirstStage():
            # Demand at period t for item i
            if forwardpass:
                righthandside = righthandside \
                                + self.SDDPOwner.CurrentSetOfTrialScenarios[scenario].Demands[
                                  self.GetTimePeriodAssociatedToInventoryVariable(p, t)][p]
            else:
                righthandside = righthandside \
                                + self.SDDPOwner.SetOfSAAScenario[self.GetTimePeriodAssociatedToInventoryVariable(p, t)][scenario][p]

        return righthandside


    def GetPIFlowFromPreviousStage(self, p, t):
        result = self.SDDPOwner.GetQuantityFixedEarlier(p, t, self.CurrentTrialNr)
        return result


    def GetFlowFromPreviousStage(self, p, t):
        result = 0
        if self.Instance.HasExternalDemand[p] and not self.IsFirstStage():
            periodbackorder = self.GetTimePeriodAssociatedToBackorderVariable(p, t) - 1
            if periodbackorder >= 0 and periodbackorder < self.TimeDecisionStage:
                # Backorder at period t-1
                result += self.SDDPOwner.GetBackorderFixedEarlier(p,
                                                                  self.GetTimePeriodAssociatedToBackorderVariable(p, t) - 1,
                                                                  self.CurrentTrialNr)

        productionstartedtime = self.GetTimePeriodAssociatedToInventoryVariable(p, t) - self.Instance.LeadTimes[p]
        if productionstartedtime >= 0 and productionstartedtime < self.TimeDecisionStage:
            # Production at period t-L_i
            result -= self.SDDPOwner.GetQuantityFixedEarlier(p, productionstartedtime,
                                                             self.CurrentTrialNr)

        periodprevinventory = self.GetTimePeriodAssociatedToInventoryVariable(p, t) - 1
        if periodprevinventory >= -1 and periodprevinventory< self.TimeDecisionStage \
                and not (self.IsFirstStage() and self.Instance.HasExternalDemand[p]):
            # inventory at period t- 1
            result -= self.SDDPOwner.GetInventoryFixedEarlier(p, periodprevinventory, self.CurrentTrialNr)
        return result



    #This funciton creates all the flow constraint
    def CreateFlowConstraints(self):
        self.FlowConstraintNR = [[["" for p in self.Instance.ProductSet]
                                      for w in self.FixedScenarioSet]
                                      for t in self.RangePeriodInv]
        for w in self.FixedScenarioSet:
            for t in self.RangePeriodInv:
                for p in self.GetProductWithStockVariable(t):
                        backordervar = []
                        consumptionvar = []
                        consumptionvarcoeff = []
                        righthandside = [self.GetRHSFlow(p, t, w, self.IsForward)]
                        if self.Instance.HasExternalDemand[p]:# and not self.IsFirstStage():
                             backordervar = [self.GetIndexBackorderVariable(p, t, w)]

                        else:
                             #dependentdemandvar = [self.GetIndexQuantityVariable(q) for q in self.Instance.RequieredProduct[p]]
                             #dependentdemandvarcoeff = [-1 * self.Instance.Requirements[q][p] for q in self.Instance.RequieredProduct[p]]
                             consumptionvar = [self.GetIndexConsumptionVariable(c, t, w)
                                                for c in self.Instance.ConsumptionSet if p == c[0]]
                             consumptionvarcoeff = [-1 for c in self.Instance.ConsumptionSet if p == c[0]]

                        inventoryvar = [self.GetIndexStockVariable(p, t, w)]

                        qtyvar = []
                        qtycoeff = []

                        productionperiod = self.GetTimePeriodAssociatedToInventoryVariable(p, t) - self.Instance.LeadTimes[p]

                        if productionperiod >= self.GetTimePeriodAssociatedToQuantityVariable(p, 0):
                            qtytimeinstage = self.GetTimeIndexForQty(p, productionperiod)
                            qtyvar.append(self.GetIndexQuantityVariable(p, qtytimeinstage, w))
                            qtycoeff.append(1)

                        previnvvar = []
                        previnvcoeff = []
                        periodflow = self.GetTimePeriodAssociatedToInventoryVariable(p, t) - 1
                        if periodflow >= 0 and periodflow >= self.GetTimePeriodAssociatedToInventoryVariable(p, 0):
                            previnvvar.append(self.GetIndexStockVariable(p, t-1, w))
                            previnvcoeff.append(1)
                            if self.Instance.HasExternalDemand[p]:# and (t >= 2 or not self.IsFirstStage()):
                                previnvvar.append(self.GetIndexBackorderVariable(p, t-1, w))
                                previnvcoeff.append(-1)

                        flowfrompreviousstagevar = [self.GetIndexFlowFromPreviousStage(p, t)]

                        vars = inventoryvar + backordervar + flowfrompreviousstagevar + consumptionvar + qtyvar + previnvvar

                        coeff = [-1] * len(inventoryvar) \
                                + [1] * len(backordervar) \
                                + [-1]  \
                                + consumptionvarcoeff \
                                + qtycoeff \
                                + previnvcoeff

                        if len(vars) > 0:
                               self.Cplex.linear_constraints.add(lin_expr=[cplex.SparsePair(vars, coeff)],
                                                                   senses=["E"],
                                                                   rhs=righthandside)
                               self.FlowConstraintNR[t][w][p] = "Flow%d_%d_%d" % (p, w,t)
                               if Constants.Debug:
                                   self.Cplex.linear_constraints.set_names(self.LastAddedConstraintIndex,
                                                                           self.FlowConstraintNR[t][w][p])
                               self.IndexFlowConstraint.append(self.LastAddedConstraintIndex)
                               self.LastAddedConstraintIndex = self.LastAddedConstraintIndex + 1
                               self.ConcernedProductFlowConstraint.append(p)
                               self.ConcernedScenarioFlowConstraint.append(w)
                               self.ConcernedTimeFlowConstraint.append(self.GetTimePeriodAssociatedToInventoryVariable(p, t))
            # This function creates all the flow constraint



    def CreatePIFlowConstraints(self):

        for w in self.FixedScenarioSet:
            for wevpi in self.EVPIScenarioRange:
                for t in self.TimePeriodToGo:
                    for p in self.Instance.ProductSet:
                        if self.IsInFutureStageForInv(p, t):
                            backordervar = []
                            consumptionvar = []
                            consumptionvarcoeff = []
                            righthandside = [self.EVPIScenarioSet[wevpi].Demands[t][p]]




                            qtyvar = []
                            qtycoeff = []
                            flowvar = []
                            flowcoeff = []
                            periodproduction = t - self.Instance.LeadTimes[p]
                            if periodproduction >= self.GetTimePeriodAssociatedToQuantityVariable(p,0):
                                qtyvar = [self.GetIndexPIQuantityVariable(p, periodproduction, wevpi, w)]
                                qtycoeff = [1.0]
                            elif periodproduction >= 0:
                                flowvar = [self.GetIndexPIFlowPrevQty(p, periodproduction)]
                                flowcoeff = [1.0]

                            if self.Instance.HasExternalDemand[p]:
                                backordervar = [self.GetIndexPIBacklogVariable(p, t, wevpi, w)]

                            else:
                                 consumptionvar = [self.GetIndexPIConsumptionVariable(c, t, wevpi, w)
                                                  for c in self.Instance.ConsumptionSet if p == c[0]]
                                 consumptionvarcoeff = [-1 for c in self.Instance.ConsumptionSet if p == c[0]]

                            inventoryvar = [self.GetIndexPIInventoryVariable(p, t, wevpi, w)]

                            previnventoryvar = []
                            previnventorycoeff = []
                            prevbackordervar = []
                            prevbackordercoeff = []
                            # consider previous inventory and backlogs
                            if t == 0:
                                righthandside[0] -= self.Instance.StartingInventories[p]
                            else:
                                previnventoryvar = [self.GetIndexPIInventoryVariable(p, t-1, wevpi, w)]
                                previnventorycoeff = [1.0]
                                if self.Instance.HasExternalDemand[p]:
                                    prevbackordervar = [self.GetIndexPIBacklogVariable(p, t-1, wevpi, w)]
                                    prevbackordercoeff = [-1.0]

                            vars = inventoryvar + backordervar + consumptionvar +\
                                   previnventoryvar + prevbackordervar + qtyvar + flowvar

                            coeff = [-1] * len(inventoryvar) \
                                    + [1] * len(backordervar) \
                                    + consumptionvarcoeff \
                                    + previnventorycoeff \
                                    + prevbackordercoeff \
                                    + qtycoeff \
                                    + flowcoeff


                            if len(vars) > 0:

                                self.Cplex.linear_constraints.add(lin_expr=[cplex.SparsePair(vars, coeff)],
                                                                  senses=["E"],
                                                                  rhs=righthandside)
                                if Constants.Debug:
                                    self.Cplex.linear_constraints.set_names(self.LastAddedConstraintIndex,
                                                                            "Flow%d" % (p))

                                self.IndexPIFlowConstraint.append(self.LastAddedConstraintIndex)
                                self.LastAddedConstraintIndex = self.LastAddedConstraintIndex + 1
                                self.ConcernedTimePIFlowConstraint.append(t)
                                self.ConcernedProductPIFlowConstraint.append(p)
                                self.ConcernedScenarioPIFlowConstraint.append(w)
                                self.ConcernedEVPIScenarioPIFlowConstraint.append(wevpi)

    # Return the set of products which are associated with stock decisions at the current stage
    def GetProductWithStockVariable(self, t):
        result = self.Instance.ProductSet
        # At the first stage, only the components are associated with a stock variable
        if self.IsFirstStage() and t == 0:
            result = self.Instance.ProductWithoutExternalDemand
        #At the last stage only the finish product have inventory variable
        if self.IsLastStage() and t == len(self.RangePeriodEndItemInv)-1:
            result = self.Instance.ProductWithExternalDemand
        return result

    # Return the set of products which are associated with backorder decisions at the current stage
    def GetProductWithBackOrderVariable(self, t):
        result = self.Instance.ProductWithExternalDemand
        # At each stage except the first, the finsih product are associated with a backorders variable
        if self.IsFirstStage() and t == 0:
            result = []
        return result

    #This function returns the right hand side of the production consraint associated with product p
    def GetProductionConstrainRHS(self, p, t):
        if self.IsFirstStage():
            righthandside = 0.0
        else:
            yvalue = self.SDDPOwner.GetSetupFixedEarlier(p, t, self.CurrentTrialNr)
            righthandside = self.GetBigMValue(p) * yvalue

        return righthandside


    # This function creates the  indicator constraint to se the production variable to 1 when a positive quantity is produce
    def CreateProductionConstraints(self):
        for w in self.FixedScenarioSet:
            for t in self.RangePeriodQty:
                for p in self.Instance.ProductSet:
                    righthandside = [0.0]
                    if self.IsFirstStage():
                        vars = [self.GetIndexQuantityVariable(p, t, w), self.GetIndexProductionVariable(p, self.GetTimePeriodAssociatedToQuantityVariable(p,t))]
                        coeff = [-1.0, self.GetBigMValue(p)]

                        # PrintConstraint( vars, coeff, righthandside )
                        self.Cplex.linear_constraints.add(lin_expr=[cplex.SparsePair(vars, coeff)],
                                                          senses=["G"],
                                                          rhs=righthandside)
                    else:
                            vars = [self.GetIndexQuantityVariable(p, t, w),
                                    self.GetIndexProductionRHS(p, t)]

                            coeff = [1.0, -1.0]
                            # PrintConstraint( vars, coeff, righthandside )
                            self.Cplex.linear_constraints.add(lin_expr=[cplex.SparsePair(vars, coeff)],
                                                                 senses=["L"],
                                                                 rhs=righthandside)
                    self.IndexProductionQuantityConstraint.append(self.LastAddedConstraintIndex)
                    self.LastAddedConstraintIndex = self.LastAddedConstraintIndex + 1

                    self.ConcernedProductProductionQuantityConstraint.append(p)
                    self.ConcernedTimeProductionQuantityConstraint.append(self.GetTimePeriodAssociatedToQuantityVariable(p,t))
                    self.ConcernedScenarioProductionQuantityConstraint.append(w)

    def CreatePIProductionConstraints(self):
        for w in self.FixedScenarioSet:
            for wevpi in self.EVPIScenarioRange:
                for t in self.TimePeriodToGo:
                    for p in self.Instance.ProductSet:
                        if t > self.GetTimePeriodAssociatedToQuantityVariable(p, self.RangePeriodQty[-1]):

                            righthandside = [0.0]
                            vars = [self.GetIndexPIQuantityVariable(p, t, wevpi, w)]

                            coeff = [1.0]
                            if self.IsFirstStage():
                                vars = vars +[self.GetIndexProductionVariable(p, t)]
                                coeff = coeff + [-1.0*self.GetBigMValue(p)]
                            else:
                                vars = vars + [self.GetIndexPIProductionRHS(p,t)]
                                coeff = coeff + [-1.0]

                            self.Cplex.linear_constraints.add(lin_expr=[cplex.SparsePair(vars, coeff)],
                                                              senses=["L"],
                                                              rhs=righthandside)
                            self.IndexPIProductionQuantityConstraint.append(self.LastAddedConstraintIndex)
                            self.LastAddedConstraintIndex = self.LastAddedConstraintIndex + 1

                            self.ConcernedProductPIProductionQuantityConstraint.append(p)
                            self.ConcernedTimePIProductionQuantityConstraint.append(t)
                            self.ConcernedScenarioPIProductionQuantityConstraint.append(w)

    def CreateConsumptionConstraints(self):
        # Capacity constraint
        for w in self.FixedScenarioSet:
            for t in self.RangePeriodQty:
                for p in self.Instance.ProductSet:
                    for k in self.Instance.ProductSet:
                        if self.Instance.Requirements[p][k] and self.Instance.IsMaterProduct(k):
                            quantityvar = [self.GetIndexQuantityVariable(p, t, w)]
                            quantityvarcoeff = [-1.0 * self.Instance.Requirements[p][k]]
                            consumptionvars = []
                            consumptionvarcoeff = []
                            for q in self.Instance.ProductSet:
                                if self.Instance.Alternates[k][q] or k == q:
                                    consumptionvars = consumptionvars + [self.GetIndexConsumptionVariable(self.Instance.GetConsumptiontuple(q, p), t, w)]
                                    consumptionvarcoeff = consumptionvarcoeff + [1.0]
                            righthandside = [0.0]

                            vars = quantityvar + consumptionvars
                            coeff = quantityvarcoeff + consumptionvarcoeff
                            self.Cplex.linear_constraints.add(lin_expr=[cplex.SparsePair(vars, coeff)],
                                                              senses=["E"],
                                                              rhs=righthandside)
                            if Constants.Debug:
                                self.Cplex.linear_constraints.set_names(self.LastAddedConstraintIndex,
                                                                     "quantityConsumption%d->%d" % (p, k))
                            self.IndexConsumptionConstraint.append(self.LastAddedConstraintIndex)
                            self.LastAddedConstraintIndex = self.LastAddedConstraintIndex + 1

    def CreatePIConsumptionConstraints(self):
        # Capacity constraint
        for w in self.FixedScenarioSet:
            for wevpi in self.EVPIScenarioRange:
                for t in self.TimePeriodToGoQty:
                    for p in self.Instance.ProductSet:
                        for k in self.Instance.ProductSet:
                            if self.Instance.Requirements[p][k] and self.Instance.IsMaterProduct(k):
                                quantityvar = [self.GetIndexPIQuantityVariable(p, t, wevpi, w)]
                                quantityvarcoeff = [-1.0 * self.Instance.Requirements[p][k]]
                                consumptionvars = []
                                consumptionvarcoeff = []
                                for q in self.Instance.ProductSet:
                                    if self.Instance.Alternates[k][q] or k == q:
                                        consumptionvars = consumptionvars + [self.GetIndexPIConsumptionVariable(self.Instance.GetConsumptiontuple(q, p), t, wevpi, w)]
                                        consumptionvarcoeff = consumptionvarcoeff + [1.0]
                                righthandside = [0.0]

                                vars = quantityvar + consumptionvars
                                coeff = quantityvarcoeff + consumptionvarcoeff
                                self.Cplex.linear_constraints.add(lin_expr=[cplex.SparsePair(vars, coeff)],
                                                                  senses=["E"],
                                                                  rhs=righthandside)
                                if Constants.Debug:
                                    self.Cplex.linear_constraints.set_names(self.LastAddedConstraintIndex,
                                                                     "quantityPIConsumption%d->%d" % (p, k))
                                self.LastAddedConstraintIndex = self.LastAddedConstraintIndex + 1


    def CreateCapacityConstraints(self):
        # Capacity constraint
        if self.Instance.NrResource > 0 and not self.IsLastStage():
            for w in self.FixedScenarioSet:
                for t in self.RangePeriodQty:
                    for k in range(self.Instance.NrResource):
                       vars = [self.GetIndexQuantityVariable(p, t, w) for p in self.Instance.ProductSet]
                       coeff = [float(self.Instance.ProcessingTime[p][k]) for p in self.Instance.ProductSet]
                       righthandside = [float(self.Instance.Capacity[k])]

                       self.Cplex.linear_constraints.add(lin_expr=[cplex.SparsePair(vars, coeff)],
                                                          senses=["L"],
                                                          rhs=righthandside)

                       self.IndexCapacityConstraint.append(self.LastAddedConstraintIndex)
                       self.LastAddedConstraintIndex = self.LastAddedConstraintIndex + 1

                       self.ConcernedResourceCapacityConstraint.append(k)
                       self.ConcernedScenarioCapacityConstraint.append(w)
                      # self.ConcernedTimeCapacityConstraint.append(t)

    def CreatePICapacityConstraints(self):
        # Capacity constraint
        if self.Instance.NrResource > 0 and not self.IsLastStage():
            for t in self.TimePeriodToGoQty:
                for w in self.FixedScenarioSet:
                    for wevpi in self.EVPIScenarioRange:
                        for k in range(self.Instance.NrResource):
                            if t > self.GetTimePeriodAssociatedToQuantityVariable(0, self.RangePeriodQty[-1]):
                                vars = [self.GetIndexPIQuantityVariable(p, t, wevpi, w) for p in self.Instance.ProductSet]
                                coeff = [float(self.Instance.ProcessingTime[p][k]) for p in self.Instance.ProductSet]
                                righthandside = [float(self.Instance.Capacity[k])]

                                self.Cplex.linear_constraints.add(lin_expr=[cplex.SparsePair(vars, coeff)],
                                                                  senses=["L"],
                                                                  rhs=righthandside)

                                self.IndexPICapacityConstraint.append(self.LastAddedConstraintIndex)
                                self.LastAddedConstraintIndex = self.LastAddedConstraintIndex + 1

                                self.ConcernedResourcePICapacityConstraint.append(k)

    def CreatePIEstiamteEVPIConstraints(self):

        for w in self.FixedScenarioSet:
            var = [self.GetIndexPIQuantityVariable(p, t, wevpi, w)
                         for p in self.Instance.ProductSet
                         for t in self.TimePeriodToGoQty
                         for wevpi in self.EVPIScenarioRange]

            coeff = [wevpi.Probability
                             * math.pow(self.Instance.Gamma, t)
                             * self.Instance.VariableCost[p]
                             if t > self.GetTimePeriodAssociatedToQuantityVariable(p, self.RangePeriodQty[-1])
                             else 0.0
                             for p in self.Instance.ProductSet
                             for t in self.TimePeriodToGoQty
                             for wevpi in self.EVPIScenarioSet]

            var = var + [self.GetIndexPIConsumptionVariable(c, t, wevpi, w)
                         for c in self.Instance.ConsumptionSet
                         for t in self.TimePeriodToGoQty
                         for wevpi in self.EVPIScenarioRange]

            coeff = coeff + [wevpi.Probability
                             * math.pow(self.Instance.Gamma, t)
                             * self.Instance.AternateCosts[c[1]][c[0]]
                             if t > self.GetTimePeriodAssociatedToQuantityVariable(p, self.RangePeriodQty[-1])
                             else 0.0
                             for c in self.Instance.ConsumptionSet
                             for t in self.TimePeriodToGoQty
                             for wevpi in self.EVPIScenarioSet]


            var = var + [self.GetIndexPIInventoryVariable(p, t, wevpi,w)
                         for p in self.Instance.ProductSet
                         for t in self.TimePeriodToGo
                         for wevpi in self.EVPIScenarioRange]


            coeff = coeff + [wevpi.Probability
                             * math.pow(self.Instance.Gamma, t)
                             * self.Instance.InventoryCosts[p]
                             if self.IsInFutureStageForInv(p, t)
                             else 0.0
                             for p in self.Instance.ProductSet
                             for t in self.TimePeriodToGo
                             for wevpi in self.EVPIScenarioSet]

            var = var + [self.GetIndexPIBacklogVariable(p, t, wevpi, w)
                         for p in self.Instance.ProductWithExternalDemand
                         for t in self.TimePeriodToGo
                         for wevpi in self.EVPIScenarioRange]

            coeff = coeff + [wevpi.Probability
                             * math.pow(self.Instance.Gamma, t)
                             * self.GetBacklogCost(p, t, wevpi, w) / self.FixedScenarioPobability[w]
                             if self.IsInFutureStageForInv(p, t)
                             else 0.0
                             for p in self.Instance.ProductWithExternalDemand
                             for t in self.TimePeriodToGo
                             for wevpi in self.EVPIScenarioSet]

            var = var + [self.GetIndexEVPICostToGo(w)]

            coeff = coeff + [-1.0]

            self.Cplex.linear_constraints.add(lin_expr=[cplex.SparsePair(var, coeff)],
                                              senses=["E"],
                                              rhs=[0.0])
            self.LastAddedConstraintIndex = self.LastAddedConstraintIndex + 1

            var = [self.GetIndexEVPICostToGo(w), self.GetIndexCostToGo(w)]
            coeff = [-1.0, 1.0]

            self.Cplex.linear_constraints.add(lin_expr=[cplex.SparsePair(var, coeff)],
                                              senses=["G"],
                                              rhs=[0.0])
            self.LastAddedConstraintIndex = self.LastAddedConstraintIndex + 1

    def GetBacklogCost(self, p, t, wevpi, w):
        if t == self.Instance.NrTimeBucket-1:
            result = wevpi.Probability * self.FixedScenarioPobability[w] \
                     * math.pow(self.Instance.Gamma, t) \
                     * self.Instance.LostSaleCost[p]
        else:
            result = wevpi.Probability * self.FixedScenarioPobability[w]  \
                    * math.pow(self.Instance.Gamma, t) \
                    * self.Instance.BackorderCosts[p]

        return result

    #Define the variables
    def DefineVariables(self):

        if Constants.Debug:
            print("period with end item inv %r"%self.RangePeriodEndItemInv)
            print("period with component inv %r"%self.RangePeriodComponentInv)
            print("period with quantities %r"%self.RangePeriodQty)
            print("Time start %r"%self.TimeDecisionStage)

        #The setups are decided at the first stage
        if self.IsFirstStage():
            if Constants.SolveRelaxationFirst:
                self.Cplex.variables.add(obj=[math.pow(self.Instance.Gamma, t)
                                                       * self.Instance.SetupCosts[p]
                                              for t in self.Instance.TimeBucketSet
                                              for p in self.Instance.ProductSet],
                                         lb=[0.0] * self.NrProductionVariable,
                                         ub=[1.0] * self.NrProductionVariable)
                                                    #types=['B'] * self.NrProductionVariable)
            else:
                self.Cplex.variables.add(obj=[math.pow(self.Instance.Gamma, t)
                                              * self.Instance.SetupCosts[p]
                                              for t in self.Instance.TimeBucketSet
                                              for p in self.Instance.ProductSet],
                                         types=['B'] * self.NrProductionVariable)


        #Variable for the production quantity
        self.Cplex.variables.add(obj=[math.pow(self.Instance.Gamma, self.GetTimePeriodAssociatedToQuantityVariable(p,t))
                                      * self.FixedScenarioPobability[w]
                                      * self.Instance.VariableCost[p]
                                      for t in self.RangePeriodQty
                                      for w in self.FixedScenarioSet
                                      for p in self.Instance.ProductSet],
                                 lb=[0.0] * self.NrQuantityVariable,
                                 ub=[self.M] * self.NrQuantityVariable)

        #Variable for the consumption
        self.Cplex.variables.add(obj=[math.pow(self.Instance.Gamma,  self.GetTimePeriodAssociatedToQuantityVariable(p, t))
                                      * self.FixedScenarioPobability[w]
                                      * self.Instance.AternateCosts[c[1]][c[0]]
                                      for t in self.RangePeriodQty
                                      for w in self.FixedScenarioSet
                                      for c in self.Instance.ConsumptionSet],
                                 lb=[0.0] * self.NrConsumptionVariable,
                                 ub=[self.M] * self.NrConsumptionVariable)

        #Variable for the inventory
        #productwithstockvariable = self.GetProductWithStockVariable(t)
        self.Cplex.variables.add(obj=[math.pow(self.Instance.Gamma, self.GetTimePeriodAssociatedToInventoryVariable(p,t))
                                      * self.FixedScenarioPobability[w]
                                      * self.Instance.InventoryCosts[p]
                                     if self.Instance.HasExternalDemand[p]
                                     else math.pow(self.Instance.Gamma, self.DecisionStage)
                                           * self.Instance.InventoryCosts[p]
                                           * self.FixedScenarioPobability[w]
                                      for t in self.RangePeriodInv
                                      for w in self.FixedScenarioSet
                                      for p in self.GetProductWithStockVariable(t)],
                                  lb=[0.0] * self.NrStockVariable,
                                  ub=[self.M] * self.NrStockVariable)

        self.ComputeStockIndexArray()

        # Backorder/lostsales variables
        self.Cplex.variables.add(obj=[math.pow(self.Instance.Gamma, self.GetTimePeriodAssociatedToBackorderVariable(p, t))
                                      * self.FixedScenarioPobability[w]
                                      * self.Instance.BackorderCosts[p]
                                      if  self.GetTimePeriodAssociatedToBackorderVariable(p, t) < self.Instance.NrTimeBucket-1
                                      else math.pow(self.Instance.Gamma, self.GetTimePeriodAssociatedToBackorderVariable(p, t))
                                           * self.Instance.LostSaleCost[p]
                                           * self.FixedScenarioPobability[w]
                                      for t in self.RangePeriodEndItemInv
                                      for w in self.FixedScenarioSet
                                      for p in self.GetProductWithBackOrderVariable(t)],
                                      lb=[0.0] * self.NrBackOrderVariable,
                                      ub=[self.M] * self.NrBackOrderVariable)

        if not self.IsLastStage():
            self.Cplex.variables.add(obj=self.FixedScenarioPobability,
                                     lb=[0.0]*self.NrCostToGo,
                                     ub=[self.M]*self.NrCostToGo)

        #Compute the Flow from previous stage
        flowfromprevioustage = [self.GetFlowFromPreviousStage(p, t)
                                for t in self.RangePeriodInv
                                for p in self.GetProductWithStockVariable(t)]

        self.ComputeFlowIndexArray()

        self.Cplex.variables.add(obj=[0.0] * self.NRFlowFromPreviousStage,
                                 lb=flowfromprevioustage,
                                 ub=flowfromprevioustage)
        if not self.IsFirstStage():
            if Constants.SDDPUseEVPI:
                productionrhs = [self.GetProductionConstrainRHS(p, t)
                                 for t in self.TimePeriodToGo
                                 for p in self.Instance.ProductSet]



            else:
                productionrhs = [self.GetProductionConstrainRHS(p, self.GetTimePeriodAssociatedToQuantityVariable(p, t))
                                 for t in self.RangePeriodQty
                                 for p in self.Instance.ProductSet]

            self.Cplex.variables.add(obj=[0.0] * self.NRProductionRHS,
                                     lb=productionrhs,
                                     ub=productionrhs)


        if Constants.SDDPUseEVPI and not self.IsLastStage():
             self.Cplex.variables.add(obj=[0.0] * self.NrPIQuantity,
                                      lb=[0.0] * self.NrPIQuantity,
                                      ub=[self.M] * self.NrPIQuantity)
             self.Cplex.variables.add(obj=[0.0] * self.NrPIInventory,
                                     lb=[0.0] * self.NrPIInventory,
                                     ub=[self.M] * self.NrPIInventory)
             self.Cplex.variables.add(obj=[0.0] * self.NrPIBacklog,
                                     lb=[0.0] * self.NrPIBacklog,
                                     ub=[self.M] * self.NrPIBacklog)
             #self.M] * self.NrPIBacklog)
             self.Cplex.variables.add(obj=[0.0] * self.NrPIConsumption,
                                     lb=[0.0] * self.NrPIConsumption,
                                     ub=[self.M] * self.NrPIConsumption)


             piflowfromprevioustage = [self.GetPIFlowFromPreviousStage(p, t)
                                        for t in range(0, self.TimeDecisionStage)
                                        for p in self.Instance.ProductSet]

             self.Cplex.variables.add(obj=[0.0] * self.NrPIFlowFromPreviouQty,
                                      lb=piflowfromprevioustage,
                                      ub=piflowfromprevioustage)

             self.Cplex.variables.add(obj=[0.0] * self.NrEstmateCostToGoEVPI,
                                     lb=[0.0] * self.NrEstmateCostToGoEVPI,
                                     ub=[self.M] * self.NrEstmateCostToGoEVPI)


        #In debug mode, the variables have names
        if Constants.Debug:
            self.AddVariableName()

    #Add the name of each variable
    def AddVariableName(self):
        if Constants.Debug:
            print("Add the names of the variable")
        # Define the variable name.
        # Usefull for debuging purpose. Otherwise, disable it, it is time consuming.
        if Constants.Debug:
            quantityvars = []
            consumptionvars = []
            inventoryvars = []
            productionvars = []
            backordervars = []
            productionrhsvar = []
            flowfrompreviousstgaevar = []
            piflow = []
            if self.IsFirstStage():
                for p in self.Instance.ProductSet:
                    for t in self.Instance.TimeBucketSet:
                        productionvars.append((self.GetIndexProductionVariable(p, t), self.GetNameProductionVariable(p, t)))
            else:
                if not self.IsLastStage():
                    timeset = self.TimePeriodToGo
                    if not Constants.SDDPUseEVPI:
                        timeset = [self.TimeDecisionStage + tau for tau in self.RangePeriodQty]
                    for t in timeset:
                        for p in self.Instance.ProductSet:
                            productionrhsvar.append((self.GetIndexPIProductionRHS(p, t), self.GetNameProductionRHS(p, t)))

            #productwithstockvariable = self.GetProductWithStockVariable()

            for t in self.RangePeriodInv:
                for p in self.GetProductWithStockVariable(t):
                    flowfrompreviousstgaevar.append((self.GetIndexFlowFromPreviousStage(p, t), self.GetNameFlowFromPreviousStage(p, t)))
            for w in self.FixedScenarioSet:
                for t in self.RangePeriodInv:
                    for p in self.GetProductWithStockVariable(t):
                        inventoryvars.append((self.GetIndexStockVariable(p, t, w), self.GetNameStockVariable(p, t, w)))

                    for p in self.GetProductWithBackOrderVariable(t):
                        inventoryvars.append((self.GetIndexBackorderVariable(p, t, w), self.GetNameBackorderVariable(p, t, w)))

                for p in self.Instance.ProductSet:
                    for t in self.RangePeriodQty:
                        quantityvars.append((self.GetIndexQuantityVariable(p, t, w), self.GetNameQuantityVariable(p, t, w)))

                for c in self.Instance.ConsumptionSet:
                    for t in self.RangePeriodQty:
                        consumptionvars.append((self.GetIndexConsumptionVariable(c, t, w), self.GetNameConsumptionVariable(c, t, w)))


            costtogovars = []
            if not self.IsLastStage():
                for w in self.FixedScenarioSet:
                    costtogovars.append((self.GetIndexCostToGo(w), "E%d_%d"%(self.TimeDecisionStage + 1, w)))

            piquantityvar = []
            pibacklogvar = []
            pistockvar = []
            piconsumptionvar = []
            picosttogovar = []
            if Constants.SDDPUseEVPI and not self.IsLastStage():
                for p in self.Instance.ProductSet:
                    for t in range(self.Instance.NrTimeBucket - self.NrTimePeriodToGo):
                        if not self.IsLastStage():
                            piflow.append((self.GetIndexPIFlowPrevQty(p, t), self.GetNamePIFlowPrevQty(p, t)))

                for w in self.FixedScenarioSet:
                    for wevpi in self.EVPIScenarioRange:
                        for t in self.TimePeriodToGo:
                            for p in self.Instance.ProductSet:
                                if not t in self.PeriodsInGlobalMIPQty:
                                    piquantityvar.append((self.GetIndexPIQuantityVariable(p, t, wevpi, w),
                                                          self.GetNamePIQuantityVariable(p, t, wevpi, w)))
                                if self.IsInFutureStageForInv(p, t):
                                    pistockvar.append((self.GetIndexPIInventoryVariable(p, t, wevpi, w),
                                                       self.GetNamePIStockVariable(p, t, wevpi, w)))
                                if self.Instance.HasExternalDemand[p] and not t in self.PeriodsInGlobalMIPEndItemInv:
                                    pibacklogvar.append((self.GetIndexPIBacklogVariable(p, t,  wevpi, w),
                                                         self.GetNamePIBacklogVariable(p, t,  wevpi, w)))
                            for c in self.Instance.ConsumptionSet:
                                if  not t in self.PeriodsInGlobalMIPQty:
                                    piconsumptionvar.append((self.GetIndexPIConsumptionVariable(c, t,  wevpi, w),
                                                            self.GetNamePIConsumptionVariable(c, t, wevpi, w)))


                picosttogovar = [(self.GetIndexEVPICostToGo(w), "evpi_cost_to_go") for w in self.FixedScenarioSet]

            quantityvars = list(set(quantityvars))
            consumptionvars = list(set(consumptionvars))
            productionvars = list(set(productionvars))
            inventoryvars = list(set(inventoryvars))
            backordervars = list(set(backordervars))
            costtogovars = list(set(costtogovars))
            piquantityvar = list(set(piquantityvar))
            pibacklogvar = list(set(pibacklogvar))
            pistockvar = list(set(pistockvar))
            piconsumptionvar = list(set(piconsumptionvar))
            picosttogovar = list(set(picosttogovar))
            productionrhsvar = list(set(productionrhsvar))
            flowfrompreviousstgaevar = list(set(flowfrompreviousstgaevar))
            piflow = list(set(piflow))
            varnames = quantityvars + consumptionvars + inventoryvars +\
                       productionvars + backordervars + costtogovars + \
                       piquantityvar + pibacklogvar + pistockvar + piconsumptionvar + \
                       picosttogovar + productionrhsvar + flowfrompreviousstgaevar + piflow
            print(varnames)
            self.Cplex.variables.set_names(varnames)

    def ComputeVariablePeriods(self):
        self.RangePeriodQty = self.GetNumberOfPeriodWithQuantity()
        self.RangePeriodEndItemInv = self.GetNumberOfPeriodWithEndItemInventory()
        self.RangePeriodComponentInv = self.GetNumberOfPeriodWithComponentInventory()
        self.RangePeriodInv = list(set().union(self.RangePeriodEndItemInv, self.RangePeriodComponentInv))

    def ComputeVariablePeriodsInLargeMIP(self):
        self.PeriodsInGlobalMIPQty = [t + self.TimeDecisionStage for t in self.RangePeriodQty]
        self.PeriodsInGlobalMIPEndItemInv = [t + self.TimeDecisionStage -1 for t in self.RangePeriodEndItemInv]
        self.PeriodsInGlobalMIPComponentInv = [t + self.TimeDecisionStage  for t in self.RangePeriodComponentInv]
        self.PeriodsInGlobalMIPInv = list(set().union(self.PeriodsInGlobalMIPEndItemInv, self.PeriodsInGlobalMIPComponentInv))

        #Create the MIP
    def DefineMIP(self):
        if Constants.Debug:
            print("Define the MIP of stage %d" % self.DecisionStage)
        self.DefineVariables()

        if not self.IsLastStage():
            self.CreateProductionConstraints()
            #self.Cplex.write("VariableOnly.lp")
            self.CreateConsumptionConstraints()
            #self.Cplex.write("VariableOnly.lp")

            self.CreateCapacityConstraints()
            #self.Cplex.write("VariableOnly.lp")

            for c in self.SDDPCuts:
                c.AddCut()

        self.CreateFlowConstraints()
        #self.Cplex.write("VariableOnly.lp")

      #  if self.IsFirstStage() and Constants.SDDPUseValidInequalities:
      #      self.CreateValideInequalities()

        if (not self.IsLastStage()) and Constants.SDDPUseEVPI:
            self.EVPIScenarioSet = self.SDDPOwner.GenerateScenarios(Constants.SDDPNrEVPIScenario, average=True)

            # set the demand to the average of the sample
            scenario = self.EVPIScenarioSet[0]
            for t in self.Instance.TimeBucketSet:
                for p in self.Instance.ProductWithExternalDemand:

                    scenario.Demands[t][p] = sum(self.SDDPOwner.SetOfSAAScenario[t][w][p]
                                                 for w in self.SDDPOwner.SAAScenarioNrSetInPeriod[t]) \
                                             / self.SDDPOwner.NrScenarioSAA
            #        for w in self.SDDPOwner.SetOfSAAScenario:
            #            print("________________________________________")
            #            print(w.Demands[t][p])

            #for t in self.Instance.TimeBucketSet:
            #    for p in self.Instance.ProductWithExternalDemand:
            #        print("demand in period %r, prod %r: %r"%(t, p, scenario.Demands[t][p]) )
            #self.Cplex.write("VariableOnly.lp")
            self.CreatePIFlowConstraints()
            #self.Cplex.write("VariableOnly.lp")
            #self.CreatePICapacityConstraints()
            self.CreatePIConsumptionConstraints()
            #self.Cplex.write("VariableOnly.lp")
            self.CreatePIProductionConstraints()
            #self.Cplex.write("VariableOnly.lp")

            self.CreatePICapacityConstraints()
            #self.Cplex.write("VariableOnly.lp")

            self.CreatePIEstiamteEVPIConstraints()
            #self.Cplex.write("VariableOnly.lp")

        self.MIPDefined = True

    def UpdateMipForTrialInBackard(self, trial):
        self.CurrentTrialNr = trial

        flowvariabletuples = [(self.GetIndexFlowFromPreviousStage(p,t), self.GetFlowFromPreviousStage(p, t))
                                 for t in self.RangePeriodInv for p in self.GetProductWithStockVariable(t)]
        self.Cplex.variables.set_lower_bounds(flowvariabletuples)
        self.Cplex.variables.set_upper_bounds(flowvariabletuples)

        constraintuples = []
        if len(self.SDDPCuts) > 0:
            cutvariabletuples = [(self.GetIndexCutRHSFromPreviousSatge(cut), cut.ComputeRHSFromPreviousStage(False))
                                      for cut in self.SDDPCuts if cut.IsActive]
            self.Cplex.variables.set_lower_bounds(cutvariabletuples)
            self.Cplex.variables.set_upper_bounds(cutvariabletuples)
            # Do not modify a cut that is not already added
           # if cut.IsActive:
           #         constraintuples = constraintuples + cut.ModifyCut(False)

        #if len(constraintuples) > 0:
        #    self.Cplex.linear_constraints.set_rhs(constraintuples)

    #The function below update the constraint of the MIP to correspond to the new scenario
    def UpdateMIPForScenarioAndTrialSolution(self, scenarionr, trial, forward):
        self.CurrentTrialNr = trial

        constraintuples = []

        flowvariabletuples = [(self.GetIndexFlowFromPreviousStage(p, t), self.GetFlowFromPreviousStage(p, t))
                              for t in self.RangePeriodInv for p in self.GetProductWithStockVariable(t)]
        self.Cplex.variables.set_lower_bounds(flowvariabletuples)
        self.Cplex.variables.set_upper_bounds(flowvariabletuples)

        if Constants.SDDPUseEVPI and not self.IsFirstStage() and not self.IsLastStage():
            flowvariabletuples = [(self.GetIndexPIFlowPrevQty(p, t), self.GetPIFlowFromPreviousStage(p, t))
                                  for p in self.Instance.ProductSet
                                  for t in range(0, self.TimeDecisionStage)]
            self.Cplex.variables.set_lower_bounds(flowvariabletuples)
            self.Cplex.variables.set_upper_bounds(flowvariabletuples)


        for i in range(len(self.IndexFlowConstraint)):
            constr = self.IndexFlowConstraint[i]
            p = self.ConcernedProductFlowConstraint[i]
            t = self.ConcernedTimeFlowConstraint[i]
            tindex = self.GetTimeIndexForInv(p, t)
            rhs = self.GetRHSFlow(p, tindex, scenarionr, forward)
            #else:
            #    t = self.ConcernedTimeFlowConstraint[i]
            #    rhs = self.GetRHSFlowConst(p, t)
            constraintuples.append((constr, rhs))

        if len(constraintuples) > 0:
           self.Cplex.linear_constraints.set_rhs(constraintuples)

        if len(self.SDDPCuts) > 0:
            cutvariabletuples = [(self.GetIndexCutRHSFromPreviousSatge(cut), cut.ComputeRHSFromPreviousStage(forward))
                                         for cut in self.SDDPCuts if cut.IsActive]
            self.Cplex.variables.set_lower_bounds(cutvariabletuples)
            self.Cplex.variables.set_upper_bounds(cutvariabletuples)
        #for cut in self.SDDPCuts:
            # Do not modify a cut that is not already added
        #    if cut.IsActive:
        #        constraintuples = constraintuples + cut.ModifyCut(forward)

        if len(constraintuples) > 0:
           self.Cplex.linear_constraints.set_rhs(constraintuples)


    #This function update the MIP for the current stage, taking into account the new value fixedin the previous stage
    def UpdateMIPForStage(self):

        if not self.IsFirstStage() and not self.IsLastStage():


            if Constants.SDDPUseEVPI:
                productionvalue = [(self.GetIndexPIProductionRHS(p, t), self.GetProductionConstrainRHS(p, t))
                                   for p in self.Instance.ProductSet for t in self.TimePeriodToGo]
                self.Cplex.variables.set_lower_bounds(productionvalue)
                self.Cplex.variables.set_upper_bounds(productionvalue)

                flowvariabletuples = [(self.GetIndexPIFlowPrevQty(p, t), self.GetPIFlowFromPreviousStage(p, t))
                                      for p in self.Instance.ProductSet
                                      for t in range(0, self.TimeDecisionStage)]
                self.Cplex.variables.set_lower_bounds(flowvariabletuples)
                self.Cplex.variables.set_upper_bounds(flowvariabletuples)

            else:
                productionvalue = [(self.GetIndexProductionRHS(p, t),
                                    self.GetProductionConstrainRHS(p, self.GetTimePeriodAssociatedToQuantityVariable(p,t)))
                                   for t in self.RangePeriodQty
                                   for p in self.Instance.ProductSet]
                self.Cplex.variables.set_lower_bounds(productionvalue)
                self.Cplex.variables.set_upper_bounds(productionvalue)

        #else:
        #    if not self.IsLastStage():
        #        raise NameError("Should not enter here")
        #        constraintuples =[]
        #       # for i in range(len(self.IndexProductionQuantityConstraint)):
        #       #         constr = self.IndexProductionQuantityConstraint[i]
        #       #         p = self.ConcernedProductProductionQuantityConstraint[i]
        #       #         if not self.IsLastStage():
        #       #             rhs = self.GetProductionConstrainRHS(p)
        #       #         else:
        #       #             t = self.ConcernedTimeProductionQuantityConstraint[i]
        #       #             rhs= self.GetProductionConstrainRHS(p,t)

        #       #         constraintuples.append((constr, rhs))

        #        for i in range(len(self.IndexProductionQuantityConstraint)):
        #            constr = self.IndexProductionQuantityConstraint[i]
        #            p = self.ConcernedProductProductionQuantityConstraint[i]
        #            rhs = self.GetProductionConstrainRHS(p)

        #            constraintuples.append((constr, rhs))

                    # self.Cplex.linear_constraints.set_rhs(constraintuples)

                    # constrainttuples = []

        #        if len(constraintuples) > 0:
                    # print constraintuples
        #            self.Cplex.linear_constraints.set_rhs(constraintuples)



    #This run the MIP of the current stage (one for each scenario)
    def RunForwardPassMIP(self):

       # generatecut = False #self.IsLastStage() and not self.SDDPOwner.EvaluationMode \
                      #and (self.SDDPOwner.UseCorePoint or not Constants.GenerateStrongCut)



        if Constants.Debug:
            print("build the MIP of stage %d" %self.DecisionStage)

        # Create a cute for the previous stage problem
        #if generatecut:
        #    cut = SDDPCut(self.PreviousSDDPStage)

        averagecostofthesubproblem = 0
        if self.MIPDefined:
            self.UpdateMIPForStage()

        if self.IsFirstStage() or self.TimeDecisionStage <= self.Instance.NrTimeBucketWithoutUncertaintyBefore:
            consideredscenario = [0]#range( len( self.SDDPOwner.CurrentSetOfScenarios ) )
        else:
            consideredscenario = range(len(self.SDDPOwner.CurrentSetOfTrialScenarios))

        for w in consideredscenario:
            if not self.MIPDefined:
                self.CurrentTrialNr = w
                self.DefineMIP()
            else:
                self.UpdateMIPForScenarioAndTrialSolution(w, w, True)

            if Constants.SDDPPrintDebugLPFiles : #or self.IsFirstStage():
                print("Update or build the MIP of stage %d for scenario %d" %(self.DecisionStage, w))
                self.Cplex.write("./Temp/stage_%d_iter_%d_scenar_%d.lp" % (self.DecisionStage, self.SDDPOwner.CurrentIteration, w))
            else:
                # name = "mrp_log%r_%r_%r" % ( self.Instance.InstanceName, self.Model, self.DemandScenarioTree.Seed )
                if not Constants.Debug:
                    self.Cplex.set_log_stream(None)
                    self.Cplex.set_results_stream(None)
                    self.Cplex.set_warning_stream(None)
                    self.Cplex.set_error_stream(None)
            self.Cplex.parameters.advance = 1
            self.Cplex.parameters.lpmethod = 2  # Dual primal cplex.CPX_ALG_DUAL
            self.Cplex.parameters.threads.set(1)
            self.Cplex.solve()
            if Constants.Debug:
                 print("Solution status:%r"%self.Cplex.solution.get_status())
                 if Constants.SDDPPrintDebugLPFiles:  # or self.IsFirstStage():
                     self.Cplex.solution.write("./Temp/Sol_stage_%d_iter_%d_scenar_%d.lp" % (self.DecisionStage, self.SDDPOwner.CurrentIteration, w))

            if self.IsFirstStage() or self.TimeDecisionStage <= self.Instance.NrTimeBucketWithoutUncertaintyBefore:
                self.CurrentTrialNr = 0
                self.SaveSolutionForScenario()
                self.CopyDecisionOfScenario0ToAllScenario()
            else:
                self.SaveSolutionForScenario()

            averagecostofthesubproblem += self.Cplex.solution.get_objective_value() \
                                          * self.SDDPOwner.CurrentSetOfTrialScenarios[w].Probability

            #update the last iteration where dual were used
            if Constants.SDDPCleanCuts and not self.IsLastStage() and len(self.IndexCutConstraint)>0:
                sol = self.Cplex.solution
                duals = sol.get_linear_slacks(self.IndexCutConstraint)
                #print("cut slack %s"%duals)
                for i in range(len(duals)):
                    if duals[i] <> 0:
                        c = self.ConcernedCutinConstraint[i]
                        c.LastIterationWithDual = self.SDDPOwner.CurrentIteration


            #if generatecut:
            #    sol = self.Cplex.solution
            #    if Constants.SDDPPrintDebugLPFiles:
            #        sol.write("./Temp/bacward_mrpsolutionstage_%d_iter_%d_scenar_%d.sol" % (self.DecisionStage, self.SDDPOwner.CurrentIteration, w))
            #    self.ImproveCutFromSolution(cut, sol)
                #print("Cut RHS after scenario %s"%cut.GetRHS())
        #if generatecut:
                # Average by the number of scenario
        #    cut.UpdateRHS()#float(len(self.SDDPOwner.CurrentSetOfScenarios)))
        #    if Constants.Debug:
        #        self.checknewcut(cut, averagecostofthesubproblem, self.PreviousSDDPStage.Cplex.solution)

        #    cut.AddCut()

        #if self.IsLastStage():
            # Average by the number of scenario
        #    cut.DivideAllCoeff(len(self.SDDPOwner.CurrentSetOfScenarios))

    def CopyDecisionOfScenario0ToAllScenario(self):
        for w2 in range(1, len(self.SDDPOwner.CurrentSetOfTrialScenarios)):
            self.CurrentTrialNr = w2
            self.StageCostPerScenarioWithCostoGo[self.CurrentTrialNr] = self.StageCostPerScenarioWithCostoGo[0]
            self.PartialCostPerScenario[self.CurrentTrialNr] = self.PartialCostPerScenario[0]
            self.QuantityValues[self.CurrentTrialNr] = self.QuantityValues[0]
            self.ConsumptionValues[self.CurrentTrialNr] = self.ConsumptionValues[0]
            self.ProductionValue[self.CurrentTrialNr] = self.ProductionValue[0]
            self.InventoryValue[self.CurrentTrialNr] = self.InventoryValue[0]
            self.BackorderValue[self.CurrentTrialNr] = self.BackorderValue[0]

    def GetVariableValue(self, sol):
       # print("Scenario is 0, because this should only be used in forward pass")
        if len(self.RangePeriodQty) > 0:
            indexarray = [self.GetIndexQuantityVariable(p, t, 0)
                          for t in self.RangePeriodQty
                          for p in self.Instance.ProductSet]
            values = sol.get_values(indexarray)
            self.QuantityValues[self.CurrentTrialNr] = [[values[t * self.Instance.NrProduct + p]
                                                         for p in self.Instance.ProductSet]
                                                         for t in self.RangePeriodQty]

            indexarray = [self.GetIndexConsumptionVariable(c, t, 0) for t in self.RangePeriodQty for c in self.Instance.ConsumptionSet ]

            values = sol.get_values(indexarray)

            self.ConsumptionValues[self.CurrentTrialNr] = [[values[t * len(self.Instance.ConsumptionSet) + c]
                                                            for c in range(len(self.Instance.ConsumptionSet))]
                                                            for t in self.RangePeriodQty]


        if self.IsFirstStage():
            indexarray = [self.GetIndexProductionVariable(p, t) for t in self.Instance.TimeBucketSet
                          for p in self.Instance.ProductSet]
            values = sol.get_values(indexarray)

            self.ProductionValue[self.CurrentTrialNr] = [[max(values[t * self.Instance.NrProduct + p], 0.0)
                                                          for p in self.Instance.ProductSet]
                                                         for t in self.Instance.TimeBucketSet]
                #[round( values[t * self.Instance.NrProduct + p], 0) for p in self.Instance.ProductSet] for t in
                #self.Instance.TimeBucketSet]

        #prductwithstock = self.GetProductWithStockVariable()
        indexarray = [self.GetIndexStockVariable(p, t, 0)  for t in self.RangePeriodInv for p in self.GetProductWithStockVariable(t)]
        values = sol.get_values(indexarray)

        self.InventoryValue[self.CurrentTrialNr] = [['nan' for p in self.Instance.ProductSet]
                                                           for t in self.RangePeriodInv]

        index = 0
        for t in self.RangePeriodInv:
            for p in self.GetProductWithStockVariable(t):
                self.InventoryValue[self.CurrentTrialNr][t][p] = values[index]
                index = index + 1

        indexarray = [self.GetIndexBackorderVariable(p, t, 0) for t in self.RangePeriodInv
                                                              for p in self.GetProductWithBackOrderVariable(t)]
        if len(indexarray)>0:
            values = sol.get_values(indexarray)

        self.BackorderValue[self.CurrentTrialNr] = [['nan' for p in self.Instance.ProductSet]
                                                    for t in self.RangePeriodInv]

        index = 0
        for t in self.RangePeriodInv:
            for p in self.GetProductWithBackOrderVariable(t):
                self.BackorderValue[self.CurrentTrialNr][t][p] = values[index]
                index = index + 1

      #  if not self.IsFirstStage():
      #      indexarray = [self.GetIndexBackorderVariable(p,0) for p in self.GetProductWithBackOrderVariable()]
      #      self.BackorderValue[self.CurrentTrialNr] = sol.get_values(indexarray)


    # This function run the MIP of the current stage
    def SaveSolutionForScenario(self):

        sol = self.Cplex.solution
        if sol.is_primal_feasible():
            if Constants.SDDPPrintDebugLPFiles:
                sol.write("./Temp/mrpsolution.sol")
            self.SaveSolutionFromSol(sol)
        else:
            self.Cplex.write("InfeasibleLP_stage_%d_iter_%d_scenar_%d.lp" % (
            self.DecisionStage, self.SDDPOwner.CurrentIteration, self.CurrentTrialNr))
            raise NameError("Infeasible sub-problem!!!!")

    def SaveSolutionFromSol(self, sol):
            obj = sol.get_objective_value()

            self.StageCostPerScenarioWithCostoGo[self.CurrentTrialNr] = obj
            self.StageCostPerScenarioWithoutCostoGo[self.CurrentTrialNr] = self.StageCostPerScenarioWithCostoGo[self.CurrentTrialNr]

            if not self.IsLastStage():
                self.StageCostPerScenarioWithoutCostoGo[self.CurrentTrialNr] = self.StageCostPerScenarioWithCostoGo[self.CurrentTrialNr] - sol.get_values([self.StartCostToGo])[0]

            if self.IsFirstStage():
                self.PartialCostPerScenario[self.CurrentTrialNr] = self.StageCostPerScenarioWithoutCostoGo[self.CurrentTrialNr]
            else:
                self.PartialCostPerScenario[self.CurrentTrialNr] = self.StageCostPerScenarioWithoutCostoGo[self.CurrentTrialNr] \
                                                                   + self.PreviousSDDPStage.PartialCostPerScenario[self.CurrentTrialNr]
            self.GetVariableValue(sol)

            if Constants.Debug:
                cotogo = 0
                if not self.IsLastStage():
                    cotogo = sol.get_values([self.StartCostToGo])[0]

                print("******************** Solution at stage %d cost: %r cost to go %r *********************"
                      %(self.DecisionStage, sol.get_objective_value(), cotogo))
                print(" Quantities: %r"%self.QuantityValues)
                if not self.IsLastStage():
                   print(" Demand: %r" %(self.SDDPOwner.CurrentSetOfTrialScenarios[self.CurrentTrialNr].Demands[self.TimeDecisionStage]))
                print(" Inventory: %r"%self.InventoryValue)
                print(" BackOrder: %r"%self.BackorderValue)
                if self.IsFirstStage():
                    print(" Production: %r "%self.ProductionValue)
                print("*************************************************************")


    def ImproveCutFromSolution(self, cut, solution):
        self.IncreaseCutWithFlowDual(cut, solution)


        if not self.IsLastStage():
            self.IncreaseCutWithProductionDual(cut, solution)
            self.IncreaseCutWithCapacityDual(cut, solution)
            self.IncreaseCutWithCutDuals(cut, solution)

            if Constants.SDDPUseEVPI:
                self.IncreaseCutWithPIFlowDual(cut, solution)

                if len(self.TimePeriodToGoQty) >= 1:
                    self.IncreaseCutWithPIProductionDual(cut, solution)
                    self.IncreaseCutWithPICapacityDual(cut, solution)


    #Generate the bender cut
    def GernerateCut(self, trial, returncut=False):
        if Constants.Debug:
            print("Generate a cut to add at stage %d" % self.PreviousSDDPStage.DecisionStage)

        if not self.IsFirstStage():
            # Re-run the MIP to take into account the just added cut
            # Solve the problem for each scenario
            self.UpdateMIPForStage()

            cutsets = []
            avgcostssubprob = []
            for trial in self.SDDPOwner.ConsideredTrialForBackward:#:
                    self.SAAStageCostPerScenarioWithoutCostoGopertrial[trial] = 0
                    self.CurrentTrialNr = trial
                    # Create a cute for the previous stage problem
                    cut = SDDPCut(self.PreviousSDDPStage, self.PreviousSDDPStage.CorrespondingForwardStage, trial)
                    averagecostofthesubproblem = 0
                #for scenario in self.SDDPOwner.SAAScenarioNrSet:
                    self.UpdateMipForTrialInBackard(trial)
                    if Constants.SDDPPrintDebugLPFiles:
                        print("Resolve for backward pass the MIP of stage %d " % (self.DecisionStage))
                        self.Cplex.write("./Temp/backward_stage_%d_iter_%d.lp"
                                         % (self.DecisionStage, self.SDDPOwner.CurrentIteration))

                    if not Constants.Debug:
                        self.Cplex.set_log_stream(None)
                        self.Cplex.set_results_stream(None)
                        self.Cplex.set_warning_stream(None)
                        self.Cplex.set_error_stream(None)

                    self.Cplex.parameters.threads.set(1)
                    self.Cplex.solve()

                    sol = self.Cplex.solution

                    if Constants.Debug:
                        print("cost of subproblem: %r"%sol.get_objective_value())
                    averagecostofthesubproblem += sol.get_objective_value() #\
                                                  #* self.SDDPOwner.SetOfSAAScenario[scenario].Probability

                    #if not self.IsLastStage():
                    #    print("uncomment if this was usefull")
                    self.SAAStageCostPerScenarioWithoutCostoGopertrial[trial] = sol.get_objective_value()
                    if not self.IsLastStage():
                        self.SAAStageCostPerScenarioWithoutCostoGopertrial[trial] -= sum(self.FixedScenarioPobability[w]
                                                                                          * sol.get_values([self.GetIndexCostToGo(w)])[0]
                                                                                         for w in self.FixedScenarioSet)

                    if Constants.SDDPPrintDebugLPFiles:
                        sol.write("./Temp/bacward_mrpsolution_stage_%d_iter_%d.sol" % (self.DecisionStage, self.SDDPOwner.CurrentIteration))

                    self.ImproveCutFromSolution(cut, sol)
                    if Constants.Debug:
                        print("cur RHS after scenario %s"%(cut.DemandAVGRHS + cut.DemandRHS + cut.CapacityRHS + cut.PreviousCutRHS + cut.InitialInventoryRHS))

                    #Average by the number of scenario
                    cut.UpdateRHS()
                    if Constants.Debug:
                       print("THERE IS NO CHECK That cuts are well generated!!!!!!!!!!!!!!")
                       #self.checknewcut(cut, averagecostofthesubproblem,  self.PreviousSDDPStage.Cplex.solution, trial)
                    cut.AddCut()
                    if Constants.SDDPPrintDebugLPFiles:
                        self.PreviousSDDPStage.Cplex.write("./Temp/PreviousstageWithCut_stage_%d_iter_%d.lp" % (self.DecisionStage, self.SDDPOwner.CurrentIteration))
                    cutsets.append(cut)
                    avgcostssubprob.append(averagecostofthesubproblem)

            return cutsets, avgcostssubprob
            #print("cut added")

    def checknewcut(self, cut, averagecostofthesubproblem, sol, trial, withcorpoint=True):

        currentosttogo = sum(self.SDDPOwner.BackwardStage[t].SAAStageCostPerScenarioWithoutCostoGopertrial[trial]
                             for t in range(self.DecisionStage, len(self.SDDPOwner.StagesSet)))

        avgcorpoint = 0
        avg = 0
        #for w in self.SDDPOwner.SAAScenarioNrSet:
            #self.PreviousSDDPStage.UpdateMIPForScenario(w)
            #self.PreviousSDDPStage.Cplex.solve()
            #print("cost from previous stage: %d" %cut.GetCostToGoLBInCUrrentSolution(self.PreviousSDDPStage.Cplex.solution))
            #cut.Print()
        avg += cut.GetCostToGoLBInCUrrentSolution(trial)
        if Constants.GenerateStrongCut and withcorpoint:
            avgcorpoint += cut.GetCostToGoLBInCorePoint(trial)

            #self.PreviousSDDPStage.Cplex.write(
            #    "./Temp/test_previous_stage_%d_iter_%d_scenar_%d.lp" % (
            #    self.DecisionStage, self.PreviousSDDPStage.DecisionStage, w))

        print("Cut added, cost to go with trial sol: %r cost to go corepoint: %r  (actual of backward pass sol: %r, avg of subproblems : %r)" % (
              avg, avgcorpoint, currentosttogo, averagecostofthesubproblem))


    def GetBigMValue(self, p):
        result = self.SDDPOwner.CurrentBigM[p]
        return result


    def IncreaseCutWithPIFlowDual(self, cut, sol):
        if Constants.Debug:
            print("Increase cut with flow dual from PI lowerbound")
        duals = sol.get_dual_values(self.IndexPIFlowConstraint)
        for i in range(len(duals)):
            scenario = self.ConcernedScenarioPIFlowConstraint[i]
            # duals[i] = duals[i] * self.SDDPOwner.SetOfSAAScenario[scenario].Probability
            if duals[i] <> 0:
                p = self.ConcernedProductPIFlowConstraint[i]
                periodproduction = self.ConcernedTimePIFlowConstraint[i] - self.Instance.LeadTimes[p]

                if periodproduction >= 0 \
                   and (len(self.PeriodsInGlobalMIPQty) == 0 \
                       or periodproduction < self.PeriodsInGlobalMIPQty[0]):
                    cut.IncreaseCoefficientQuantity(p, periodproduction, duals[i])

                periodpreviousstock = self.ConcernedTimePIFlowConstraint[i] - 1

                if periodpreviousstock >= 0 and periodpreviousstock < self.GetTimePeriodRangeForInventoryVariable(p)[0]:
                    cut.IncreaseCoefficientInventory(p, periodpreviousstock, duals[i])

                    if self.Instance.HasExternalDemand[p]:
                        cut.IncreaseCoefficientBackorder(p, periodpreviousstock, -duals[i])

                if self.Instance.HasExternalDemand[p]:
                    wevpi = self.ConcernedEVPIScenarioPIFlowConstraint[i]
                    cut.IncreaseAvgDemandRHS(duals[i]
                                             * self.EVPIScenarioSet[wevpi].Demands[self.ConcernedTimePIFlowConstraint[i]][p])

    def IncreaseCutWithFlowDual(self, cut, sol):
        if Constants.Debug:
            print("Increase cut with flow dual")
        duals = sol.get_dual_values(self.IndexFlowConstraint)

        #print("DUALS flow::::%s" % duals)

        for i in range(len(duals)):
            scenario = self.ConcernedScenarioFlowConstraint[i]
            #duals[i] = duals[i] * self.SDDPOwner.SetOfSAAScenario[scenario].Probability
            if duals[i] <> 0:
                p = self.ConcernedProductFlowConstraint[i]
                periodproduction = self.ConcernedTimeFlowConstraint[i] - self.Instance.LeadTimes[p]

                if periodproduction >= 0:
                    cut.IncreaseCoefficientQuantity(p, periodproduction, duals[i])

                periodpreviousstock = self.ConcernedTimeFlowConstraint[i] - 1

                if periodpreviousstock >= 0:
                    cut.IncreaseCoefficientInventory(p, periodpreviousstock, duals[i])
                else:
                    cut.IncreaseInitInventryRHS(-1 * duals[i] * self.Instance.StartingInventories[p])

                if self.Instance.HasExternalDemand[p]:
                    cut.IncreaseCoefficientBackorder(p, periodpreviousstock, -duals[i])
                    cut.IncreaseDemandRHS(duals[i]
                                          * self.SDDPOwner.SetOfSAAScenario[self.ConcernedTimeFlowConstraint[i]][scenario][p])



    def IncreaseCutWithPIProductionDual(self, cut, sol):
        if Constants.Debug:
                print("Increase cut with production dual")
        duals = sol.get_dual_values(self.IndexPIProductionQuantityConstraint)
        #print("Duals Production:::%s"%duals)
        for i in range(len(duals)):
            if duals[i] <> 0:
               #duals[i] = duals[i] * self.SDDPOwner.SetOfSAAScenario[scenario].Probability
                p = self.ConcernedProductPIProductionQuantityConstraint[i]
                t = self.ConcernedTimePIProductionQuantityConstraint[i]
                cut.IncreaseCoefficientProduction(p, t, -1*self.GetBigMValue(p) * duals[i])



    def IncreaseCutWithProductionDual(self, cut, sol):
        if Constants.Debug:
                print("Increase cut with production dual")
        duals = sol.get_dual_values(self.IndexProductionQuantityConstraint)
        #print("Duals Production:::%s"%duals)
        for i in range(len(duals)):
            if duals[i] <> 0:
                #duals[i] = duals[i] * self.SDDPOwner.SetOfSAAScenario[scenario].Probability
                p = self.ConcernedProductProductionQuantityConstraint[i]#self.IndexProductionQuantityConstraint[i]]
                t = self.ConcernedTimeProductionQuantityConstraint[i]#self.IndexProductionQuantityConstraint[i]]
                cut.IncreaseCoefficientProduction(p, t, -1*self.GetBigMValue(p) * duals[i])

    def IncreaseCutWithPICapacityDual(self, cut, sol):
        if Constants.Debug:
            print("Increase cut with capacity dual")
        duals = sol.get_dual_values(self.IndexPICapacityConstraint)
        for i in range(len(duals)):
            if duals[i] <> 0:
                # duals[i] = duals[i] * self.SDDPOwner.SetOfSAAScenario[scenario].Probability
                k = self.ConcernedResourcePICapacityConstraint[i]
                cut.IncreaseCapacityRHS(self.Instance.Capacity[k] * duals[i])

    def IncreaseCutWithCapacityDual(self, cut, sol):
         if Constants.Debug:
            print("Increase cut with capacity dual")
         duals = sol.get_dual_values(self.IndexCapacityConstraint)
         for i in range(len(duals)):
             if duals[i] <> 0:
                 scenario = self.ConcernedScenarioCapacityConstraint[i]
                 #duals[i] = duals[i] * self.SDDPOwner.SetOfSAAScenario[scenario].Probability
                 k = self.ConcernedResourceCapacityConstraint[i]
                 cut.IncreaseCapacityRHS(self.Instance.Capacity[k] * duals[i])

    def IncreaseCutWithCutDuals(self, cut, sol):

        #if self.SDDPOwner.CurrentIteration > 0 :
            if Constants.Debug:
                print("Increase cut with cut duals")
            duals = sol.get_dual_values(self.IndexCutConstraint)
            for i in range(len(duals)):
                if duals[i] <> 0:
                    #duals[i] = duals[i] * self.SDDPOwner.SetOfSAAScenario[scenario].Probability
                    c = self.ConcernedCutinConstraint[i]

                    #In the new cut the contribution of C to the RHS is the RHS of C plus the value of of te variable of the current stage.
                    #variableatstage = c.GetCutVariablesAtStage()
                    #valueofvariable = sol.get_values(variableatstage)
                    #coefficientvariableatstage =c.GetCutVariablesCoefficientAtStage()
                    #valueofvarsinconsraint = sum(i[0] * i[1] for i in zip(valueofvariable, coefficientvariableatstage))
                    cut.IncreasePReviousCutRHS(c.GetRHS() * duals[i])#( c.GetRHS() + valueofvarsinconsraint )* duals[i])

                    for tuple in c.NonZeroFixedEarlierProductionVar:
                        p = tuple[0]; t = tuple[1]
                        cut.IncreaseCoefficientProduction(p, t, c.CoefficientProductionVariable[t][p] * duals[i])
                    for tuple in c.NonZeroFixedEarlierQuantityVar:
                        p = tuple[0];                t = tuple[1]
                        cut.IncreaseCoefficientQuantity(p, t, c.CoefficientQuantityVariable[t][p] * duals[i])
                    for tuple in c.NonZeroFixedEarlierBackOrderVar:
                        p = tuple[0];                t = tuple[1]
                        cut.IncreaseCoefficientBackorder(p, t, c.CoefficientBackorderyVariable[t][p] * duals[i])
                    for tuple in c.NonZeroFixedEarlierStockVar:
                        p = tuple[0];                t = tuple[1]
                        cut.IncreaseCoefficientInventory(p, t, c.CoefficientStockVariable[t][p] * duals[i])

    # Try to use the corepoint method of papadakos, remove if it doesn't work
    # average current solution with last core point
    def UpdateCorePoint(self):

        if len(self.CorePointQuantityValues) >= self.SDDPOwner.CurrentForwardSampleSize:
            # Try to use the corepoint method of papadakos, remove if it doesn't work
            if not self.IsLastStage():

                self.CorePointQuantityValues = [[[0.5 * self.QuantityValues[w][t][p] + 0.5 * self.CorePointQuantityValues[w][t][p]
                                                  for p in self.Instance.ProductSet]
                                                  for t in self.RangePeriodQty]
                                                  for w in self.TrialScenarioNrSet]
            if self.IsFirstStage():
                self.CorePointProductionValue = [[[max(0.5 * self.ProductionValue[w][t][p] + 0.5 * self.CorePointProductionValue[w][t][p], 0.0)
                                                 for p in self.Instance.ProductSet]
                                                  for t in self.Instance.TimeBucketSet]
                                                 for w in self.TrialScenarioNrSet]
            # The value of the inventory variables (filled after having solve the MIPs for all scenario)
            for w in self.TrialScenarioNrSet:
                for t in self.RangePeriodInv:
                    for p in self.GetProductWithStockVariable(t):
                        self.CorePointInventoryValue[w][t][p] = 0.5 * self.InventoryValue[w][t][p] + 0.5 * self.CorePointInventoryValue[w][t][p]

            # The value of the backorder variable (filled after having solve the MIPs for all scenario)
            for w in self.TrialScenarioNrSet:
                for t in self.RangePeriodInv:
                    for p in self.GetProductWithBackOrderVariable(t):
                        self.CorePointBackorderValue[w][t][p] = 0.5 * self.BackorderValue[w][t][p] \
                                                                + 0.5 * self.CorePointBackorderValue[w][t][p]

        else:
            # Try to use the corepoint method of papadakos, remove if it doesn't work
            if not self.IsLastStage():

                self.CorePointQuantityValues = [[[self.QuantityValues[w][t][p]
                                                  for p in self.Instance.ProductSet]
                                                  for t in self.RangePeriodQty]
                                                  for w in self.TrialScenarioNrSet]
            if self.IsFirstStage():
                self.CorePointProductionValue = [[[self.ProductionValue[w][t][p]
                                                   for p in self.Instance.ProductSet]
                                                  for t in self.Instance.TimeBucketSet]
                                                 for w in self.TrialScenarioNrSet]
            # The value of the inventory variables (filled after having solve the MIPs for all scenario)
            self.CorePointInventoryValue = [[[self.InventoryValue[w][t][p]
                                             for p in self.Instance.ProductSet]
                                             for t in self.RangePeriodInv]
                                             for w in self.TrialScenarioNrSet]
            # The value of the backorder variable (filled after having solve the MIPs for all scenario)
            self.CorePointBackorderValue = [[[ self.BackorderValue[w][t][p]
                                               for p in self.Instance.ProductSet]
                                               for t in self.RangePeriodInv]
                                               for w in self.TrialScenarioNrSet]

    def ChangeSetupToBinary(self):

        indexarray = [self.GetIndexProductionVariable(p, t)
                      for t in self.Instance.TimeBucketSet
                      for p in self.Instance.ProductSet]
        self.Cplex.variables.set_types(zip(indexarray, ["B"]*len(indexarray)))
        self.Cplex.variables.set_lower_bounds(zip(indexarray, [0.0]*len(indexarray)))
        self.Cplex.variables.set_upper_bounds(zip(indexarray, [1.0]*len(indexarray)))

    def ChangeSetupToValueOfTwoStage(self):
        lbtuple=[]
        ubtuples=[]
        for p in self.Instance.ProductSet:
            for t in self.Instance.TimeBucketSet:
                lbtuple.append((self.GetIndexProductionVariable(p, t), self.SDDPOwner.HeuristicSetupValue[t][p]))
                ubtuples.append((self.GetIndexProductionVariable(p, t), self.SDDPOwner.HeuristicSetupValue[t][p]))

        self.Cplex.variables.set_lower_bounds(lbtuple)
        self.Cplex.variables.set_upper_bounds(ubtuples)


    def CleanCuts(self):
        for cut in self.ConcernedCutinConstraint:
            if cut.LastIterationWithDual < self.SDDPOwner.CurrentIteration - 50:
                cut.RemoveTheCut()