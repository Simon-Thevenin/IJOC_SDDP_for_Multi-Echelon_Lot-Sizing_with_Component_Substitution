from __future__ import absolute_import, division, print_function

from Constants import Constants
from Solution import Solution
from SDDPStage import SDDPStage
from SDDPLastStage import SDDPLastStage
from ScenarioTree import ScenarioTree
from MIPSolver import MIPSolver
from SDDPCallBack import SDDPCallBack
from ScenarioTreeNode import ScenarioTreeNode
from Scenario  import Scenario
import pickle
from SDDPUserCutCallBack import SDDPUserCutCallBack

import numpy as np
import math
import time
import random
import copy
import cplex

# This class contains the attributes and methods allowing to define the SDDP algorithm.
class SDDP(object):

    #return the object stage associated with the decision stage given in paramter
    def GetSDDPStage(self, decisionstage):
        result = None
        if decisionstage >= 0:
            result = self.Stage[decisionstage]
        return result

    #Fill the links predecessor and next of each object stage
    def LinkStages(self):
        previousstage = None
        for stage in self.ForwardStage:
            stage.PreviousSDDPStage = previousstage
            if not previousstage is None:
                previousstage.NextSDDPStage = stage
            previousstage = stage

        for stage in self.BackwardStage:
            stage.PreviousSDDPStage = previousstage
            if not previousstage is None:
                previousstage.NextSDDPStage = stage
            previousstage = stage

        curenttime = 0
        stageset = self.BackwardStage
        stageset = stageset + self.ForwardStage
        #list(set().union(self.BackwardStage, self.ForwardStage))
        for stage in stageset:
            stage.ComputeVariablePeriods()

            if stage.IsFirstStage():
                stage.TimeDecisionStage = 0
            else:
                prevstage = stage.PreviousSDDPStage
                stage.TimeDecisionStage = prevstage.TimeDecisionStage + len(prevstage.RangePeriodQty)

            if stage.DecisionStage == 1:
                stage.TimeObservationStage = self.Instance.NrTimeBucketWithoutUncertaintyBefore
            if stage.DecisionStage >= 2:
                stage.TimeObservationStage = prevstage.TimeObservationStage + len(prevstage.RangePeriodEndItemInv)

            if stage.TimeDecisionStage + max(len(stage.RangePeriodQty),1) <= self.Instance.NrTimeBucketWithoutUncertaintyBefore +1:
                stage.FixedScenarioSet = [0]
            stage.ComputeVariableIndices()
            stage.ComputeVariablePeriodsInLargeMIP()

        self.AssociateDecisionAndStages()

    def AssociateDecisionAndStages(self):
        self.ForwardStageWithBackOrderDec = [None for t in self.Instance.TimeBucketSet]
        for stage in self.ForwardStage:
            for tau in stage.RangePeriodEndItemInv:
                self.ForwardStageWithBackOrderDec[tau + stage.TimeObservationStage] = stage

        self.ForwardStageWithQuantityDec = [None for t in self.Instance.TimeBucketSet]
        for stage in self.ForwardStage:
            for tau in stage.RangePeriodQty:
                self.ForwardStageWithQuantityDec[tau + stage.TimeDecisionStage] = stage

        self.ForwardStageWithCompInvDec = [None for t in self.Instance.TimeBucketSet]
        for stage in self.ForwardStage:
            for tau in stage.RangePeriodComponentInv:
                self.ForwardStageWithCompInvDec[tau + stage.TimeDecisionStage] = stage

        # curenttime = 0
        # for stage in self.ForwardStage:
        #     stage.TimeDecisionStage = curenttime
        #     stage.ComputeVariablePeriods()
        #     stage.ComputeVariableIndices()
        #     curenttime += len(stage.RangePeriodQty)

    def __init__(self, instance, testidentifier, treestructure):
        self.Instance = instance
        self.TestIdentifier = testidentifier
        if self.TestIdentifier.SDDPSetting == "SingleCut":
            Constants.SDDPUseMultiCut = False

        if self.TestIdentifier.SDDPSetting == "NoStrongCut":
            Constants.GenerateStrongCut = False

        if self.TestIdentifier.SDDPSetting == "NoEVPI":
            Constants.SDDPUseEVPI = False

        self.MaxNrStage = 30#self.Instance.NrTimeBucketWithoutUncertainty
        nrstage = min(self.Instance.NrTimeBucket - self.Instance.NrTimeBucketWithoutUncertainty, self.MaxNrStage-1) #
        self.StagesSet = range(nrstage + 1)
        self.CurrentIteration = 0
        self.CurrentLowerBound = 0
        self.BestUpperBound = Constants.Infinity
        self.LastExpectedCostComputedOnAllScenario = Constants.Infinity
        self.CurrentBestSetups = []
        self.CurrentUpperBound = Constants.Infinity
        self.VarianceForwardPass = -1
        self.StartOfAlsorithm = time.time()
        self.CurrentSetOfTrialScenarios = []
        self.SetOfSAAScenario = []
        self.SAAScenarioNrSet = []
        self.TrialScenarioNrSet = []
        self.IsIterationWithConvergenceTest = False
        self.CurrentScenarioSeed = int(self.TestIdentifier.ScenarioSeed)
        self.StartingSeed = self.TestIdentifier.ScenarioSeed
      #  self.NrScenarioSAA = int(self.TestIdentifier.NrScenario)

        self.NrSAAScenarioInPeriod = treestructure[1:-1]
        self.ForwardStage = [SDDPStage(owner=self, decisionstage=t, fixedccenarioset=[0], isforward=True, futurscenarioset =range(self.NrSAAScenarioInPeriod[t])) for t in range(nrstage)] \
                             + [SDDPLastStage(owner=self, decisionstage=nrstage, fixedccenarioset=[0], isforward=True)]

        backwardstagescenarioset = [[0]] + [range(self.NrSAAScenarioInPeriod[t-1 + self.Instance.NrTimeBucketWithoutUncertaintyBefore]) for t in range(1,nrstage+1)]
        self.BackwardStage = [SDDPStage(owner=self, decisionstage=t, fixedccenarioset=backwardstagescenarioset[t], forwardstage=self.ForwardStage[t], isforward=False ,  futurscenarioset = range(self.NrSAAScenarioInPeriod[t])) for t in range(nrstage)] \
                             + [SDDPLastStage(owner=self, decisionstage=nrstage, fixedccenarioset= backwardstagescenarioset[nrstage], forwardstage=self.ForwardStage[nrstage], isforward=False)]

        self.DefineBakwarMip = False
        self.LinkStages()


        self.CurrentNrScenario = self.TestIdentifier.NrScenarioForward# Constants.SDDPNrScenarioForwardPass

        self.SDDPNrScenarioTest = Constants.SDDPInitNrScenarioTest
        self.CurrentForwardSampleSize = self.TestIdentifier.NrScenarioForward
        self.CurrentBigM = []
        self.ScenarioGenerationMethod = self.TestIdentifier.ScenarioSampling
        self.CurrentExpvalueUpperBound = Constants.Infinity
        self.EvaluationMode = False
        self.UseCorePoint = False
        self.GenerateStrongCut = Constants.GenerateStrongCut
        self.TraceFile = None
        self.TraceFileName = "./Temp/SDDPtrace_%s.txt" % (self.TestIdentifier.GetAsString())
        self.HeuristicSetupValue = []
        self.LastIterationWithTest = 0

        self.TimeBackward = 0
        self.TimeForwardTest = 0
        self.TimeForwardNonTest = 0

        self.NrIterationWithoutLBImprovment = 0
        self.SingleTreeCplexGap = -1

        self.CurrentSetups = []
        self.HasFixedSetup = False

        self.IterationSetupFixed = 0
        self.CurrentToleranceForSameLB = 0.00001



    #This function make the forward pass of SDDP
    def ForwardPass(self, ignorefirststage=False):
        start = time.time()
        if Constants.Debug:
            print("Start forward pass")
        for t in self.StagesSet:
            if not ignorefirststage or t >= 1:
                if Constants.SDDPCleanCuts \
                    and self.CurrentIteration > 0 \
                    and self.CurrentIteration % 100 == 0 \
                    and (t >= 1 or not Constants.SDDPRunSigleTree):
                        self.ForwardStage[t].CleanCuts()
                        self.BackwardStage[t].CleanCuts()
                        print("Clean cut Should not be used")

                #Run the forward pass at each stage t

                self.ForwardStage[t].RunForwardPassMIP()

        end = time.time()
        duration = end-start

        if self.IsIterationWithConvergenceTest:
            self.TimeForwardTest += duration
        else:
            self.TimeForwardNonTest += duration

    #This function make the backward pass of SDDP
    def BackwardPass(self, returnfirststagecut=False):
        start = time.time()
        if Constants.Debug:
            print("Start Backward pass")

        self.UseCorePoint = self.GenerateStrongCut

        for t in self.StagesSet:
            if self.GenerateStrongCut:
                self.ForwardStage[t].UpdateCorePoint()

        self.ConsideredTrialForBackward = self.TrialScenarioNrSet

        if not self.DefineBakwarMip:
            for stage in self.StagesSet:
                if not self.BackwardStage[stage].IsFirstStage():
                    self.BackwardStage[stage].CurrentTrialNr = 0
                    self.BackwardStage[stage].DefineMIP()
            self.DefineBakwarMip = True

        if Constants.SDDPPrintDebugLPFiles:  # or self.IsFirstStage():
            for stage in self.StagesSet:
                self.BackwardStage[stage].Cplex.write(
                    "./Temp/Backwardstage_%d.lp" % (self.BackwardStage[stage].DecisionStage))

        #generate a cut for each trial solution
        for t in reversed(range(1, len(self.StagesSet))):
            #Build or update the MIP of stage t
            returncutinsteadofadd = (returnfirststagecut and t == 1)
            firststagecuts, avgsubprobcosts = self.BackwardStage[t].GernerateCut(returncutinsteadofadd)



        self.UseCorePoint = False

        end = time.time()
        self.TimeBackward += (end - start)

        if returnfirststagecut:
            return firststagecuts, avgsubprobcosts


    def GenerateScenarios(self, nrscenario, average = False):
        if Constants.Debug:
            print("Start generation of new scenarios")

        #Generate a scenario tree
        treestructure = [1, nrscenario] + [1] * (self.Instance.NrTimeBucket - 1) + [0]

        scenariotree = ScenarioTree(self.Instance, treestructure, self.CurrentScenarioSeed,
                                    scenariogenerationmethod=self.ScenarioGenerationMethod,
                                    averagescenariotree=average, model=Constants.ModelYQFix)



        #Get the set of scenarios
        scenarioset = scenariotree.GetAllScenarios(computeindex=False)
        return scenarioset

    def GenerateSymetricTree(self, nrscenario, average=False):
            if Constants.Debug:
                print("Start generation of new scenarios")

            # Generate a scenario tree
            treestructure = [1] + [1] * self.Instance.NrTimeBucketWithoutUncertaintyBefore \
                            + [nrscenario] * (self.Instance.NrTimeBucket - self.Instance.NrTimeBucketWithoutUncertaintyBefore) \
                            + [0]

            self.SAAScenarioTree = ScenarioTree(self.Instance, treestructure, self.CurrentScenarioSeed,
                                        scenariogenerationmethod=self.ScenarioGenerationMethod,
                                        averagescenariotree=average, model=Constants.ModelYFix,
                                        issymetric=True)

            self.CompleteSetOfSAAScenario = self.SAAScenarioTree.GetAllScenarios(computeindex=False)

            self.NrSAAScenarioInPeriod = [ 1
                                           if t < self.Instance.NrTimeBucketWithoutUncertaintyBefore
                                           else
                                           nrscenario
                                           for t in self.Instance.TimeBucketSet]
            self.SAAScenarioNrSetInPeriod = [range(self.NrSAAScenarioInPeriod[t]) for t in self.Instance.TimeBucketSet]

            saascenario = [[[] for w in self.SAAScenarioNrSetInPeriod[t]] for t in self.Instance.TimeBucketSet]
            saascenarioproba = [[-1 for w in self.SAAScenarioNrSetInPeriod[t]] for t in self.Instance.TimeBucketSet]

            t = 0
            rootofcurrentsubtree = self.SAAScenarioTree.RootNode.Branches[0]
            while len(rootofcurrentsubtree.Branches) > 0:
                for w in range(len(rootofcurrentsubtree.Branches)):
                    saascenario[t][w] = rootofcurrentsubtree.Branches[w].Demand
                    saascenarioproba[t][w] = rootofcurrentsubtree.Branches[w].Probability
                rootofcurrentsubtree = rootofcurrentsubtree.Branches[0]
                t = t + 1

            return saascenario, saascenarioproba




        #This function generates the scenarios for the current iteration of the algorithm
    def GenerateTrialScenarios(self):
        if self.IsIterationWithConvergenceTest:
            self.CurrentNrScenario = self.SDDPNrScenarioTest
        else:
            self.CurrentNrScenario = self.CurrentForwardSampleSize

        if Constants.SDDPForwardPassInSAATree:
            self.CurrentSetOfTrialScenarios =[]
            for w in range(self.CurrentNrScenario):
                    selected = self.CreateRandomScenarioFromSAA()
                    self.CurrentSetOfTrialScenarios.append(selected)

            if self.IsIterationWithConvergenceTest and Constants.MIPBasedOnSymetricTree:
                self.CurrentSetOfTrialScenarios = self.CreateAllScenarioFromSAA()

            self.TrialScenarioNrSet = range(len(self.CurrentSetOfTrialScenarios))
            self.CurrentNrScenario = len(self.CurrentSetOfTrialScenarios)
           # self.SDDPNrScenarioTest = self.CurrentNrScenario
        else:
            self.CurrentSetOfTrialScenarios = self.GenerateScenarios(self.CurrentNrScenario, average=Constants.SDDPDebugSolveAverage)
            self.TrialScenarioNrSet = range(len(self.CurrentSetOfTrialScenarios))
            self.CurrentNrScenario = len(self.CurrentSetOfTrialScenarios)
            #Modify the number of scenario at each stage
        for stage in self.StagesSet:
            self.ForwardStage[stage].SetNrTrialScenario(self.CurrentNrScenario)
            self.ForwardStage[stage].FixedScenarioPobability = [1]
            self.BackwardStage[stage].SAAStageCostPerScenarioWithoutCostoGopertrial = [0 for w in self.TrialScenarioNrSet]
            self.BackwardStage[stage].CurrentTrialNr = 0
        self.CurrentScenarioSeed = self.CurrentScenarioSeed + 1


    #This function generates the scenarios for the current iteration of the algorithm
    def GenerateSAAScenarios(self):

        #self.SetOfSAAScenario = self.GenerateScenarios(self.NrScenarioSAA, average=Constants.SDDPDebugSolveAverage)
        self.SetOfSAAScenario, saascnarioproba = self.GenerateSymetricTree(self.NrScenarioSAA, average=Constants.SDDPDebugSolveAverage)

        #self.NrScenarioSAA = len(self.SetOfSAAScenario)
        #Modify the number of scenario at each stage
        for stage in self.StagesSet:
            time = self.BackwardStage[stage].TimeDecisionStage -1

            if time + max(len(self.BackwardStage[stage].RangePeriodQty),1) >= self.Instance.NrTimeBucketWithoutUncertaintyBefore + 1:
                self.BackwardStage[stage].FixedScenarioSet = self.SAAScenarioNrSetInPeriod[time]
                self.BackwardStage[stage].FixedScenarioPobability = [saascnarioproba[time][w] for w in self.SAAScenarioNrSetInPeriod[time]]
                self.BackwardStage[stage].SAAStageCostPerScenarioWithoutCostoGopertrial = [0 for w in self.TrialScenarioNrSet]
            else:
                self.BackwardStage[stage].FixedScenarioSet = [0]
                self.BackwardStage[stage].FixedScenarioPobability = [1]
                self.BackwardStage[stage].SAAStageCostPerScenarioWithoutCostoGopertrial = [0 for w in
                                                                                           self.TrialScenarioNrSet]


        self.CurrentScenarioSeed = self.CurrentScenarioSeed + 1

        self.SetCurrentBigM()

    def GenerateSAAScenarios2(self):

         # self.NrSAAScenarioInPeriod = [1
         #                              if t < self.Instance.NrTimeBucketWithoutUncertaintyBefore
         #                              else
         #                              self.NrScenarioSAA
         #                              for t in self.Instance.TimeBucketSet]
         #
         # print("Attention hardcoded shit")
         # for t in range(self.Instance.NrTimeBucketWithoutUncertaintyBefore+3, self.Instance.NrTimeBucket):
         #     self.NrSAAScenarioInPeriod[t] = 5


         self.SAAScenarioNrSetInPeriod = [range(self.NrSAAScenarioInPeriod[t]) for t in self.Instance.TimeBucketSet]

         SymetricDemand = [[] for t in self.Instance.TimeBucketSet]
         SymetricProba = [[] for t in self.Instance.TimeBucketSet]
         np.random.seed(self.TestIdentifier.ScenarioSeed)

         for t in self.Instance.TimeBucketSet:
              SymetricDemand[t], SymetricProba[t] = \
              ScenarioTreeNode.CreateDemandNormalDistributiondemand(self.Instance, t,  self.NrSAAScenarioInPeriod[t],
                                                                    t < self.Instance.NrTimeBucketWithoutUncertaintyBefore,
                                                                    self.ScenarioGenerationMethod)

         self.SetOfSAAScenario = [[[] for w in self.SAAScenarioNrSetInPeriod[t]] for t in self.Instance.TimeBucketSet]
         self.Saascenarioproba = [[-1 for w in self.SAAScenarioNrSetInPeriod[t]] for t in self.Instance.TimeBucketSet]

         t = 0
         for t in self.Instance.TimeBucketSet:
             for w in self.SAAScenarioNrSetInPeriod[t]:
                 self.SetOfSAAScenario[t][w] = [SymetricDemand[t][p][w] for p in self.Instance.ProductSet]
                 self.Saascenarioproba[t][w] = SymetricProba[t][w]

         for stage in self.StagesSet:
            time = self.BackwardStage[stage].TimeDecisionStage -1

            if time + max(len(self.BackwardStage[stage].RangePeriodQty),1) >= self.Instance.NrTimeBucketWithoutUncertaintyBefore + 1:
                self.BackwardStage[stage].FixedScenarioSet = self.SAAScenarioNrSetInPeriod[time]

                self.BackwardStage[stage].FixedScenarioPobability = [ self.Saascenarioproba[time][w] for w in self.SAAScenarioNrSetInPeriod[time]]
                self.BackwardStage[stage].SAAStageCostPerScenarioWithoutCostoGopertrial = [0 for w in self.TrialScenarioNrSet]
            else:
                self.BackwardStage[stage].FixedScenarioSet = [0]
                self.BackwardStage[stage].FixedScenarioPobability = [1]
                self.BackwardStage[stage].SAAStageCostPerScenarioWithoutCostoGopertrial = [0 for w in self.TrialScenarioNrSet]
            for stage in self.StagesSet:
                if not self.BackwardStage[stage].IsLastStage():
                    nextstage = stage + 1
                    nextstagetime = self.BackwardStage[nextstage].TimeDecisionStage - 1
                    self.BackwardStage[stage].FuturScenarProba = self.BackwardStage[
                        nextstage].FixedScenarioPobability  # FixedScenarioPobability [ self.Saascenarioproba[nextstagetime][w] for w in self.SAAScenarioNrSetInPeriod[nextstagetime]]
                    self.BackwardStage[stage].FuturScenario = self.BackwardStage[
                        nextstage].FixedScenarioSet  # self.SAAScenarioNrSetInPeriod[nextstagetime]
                    self.ForwardStage[stage].FuturScenarProba = self.BackwardStage[
                        nextstage].FixedScenarioPobability  # [ self.Saascenarioproba[nextstagetime][w] for w in self.SAAScenarioNrSetInPeriod[nextstagetime]]
                    self.ForwardStage[stage].FuturScenario = self.BackwardStage[nextstage].FixedScenarioSet

                self.ForwardStage[stage].ComputeVariableIndices()
                self.BackwardStage[stage].ComputeVariableIndices()
            #    self.BackwardStage[stage].FixedScenarioSet = [0]
            #    self.BackwardStage[stage].FixedScenarioPobability = [1]
         #Compute the value of big M
         #create fictif the scenario with worse demand for all product
         w = Scenario()

         w.Demands = [[max(self.SetOfSAAScenario[t][w][p]  for w in self.SAAScenarioNrSetInPeriod[t])
                         for p in self.Instance.ProductSet] for t in self.Instance.TimeBucketSet]

         self.CurrentBigM = [MIPSolver.GetBigMValue(self.Instance,[w], p) for p in
                             self.Instance.ProductSet]




    def CreateRandomScenarioFromSAA(self):
        w = Scenario( proabability=1)

        w.Demands = [[] for t in self.Instance.TimeBucketSet]
        for s in range(len(self.StagesSet)):# in self.Instance.TimeBucketSet:

            if self.ForwardStage[s].TimeObservationStage >= 0:
                x = random.randint(0, self.NrSAAScenarioInPeriod[self.ForwardStage[s].TimeObservationStage]-1)
            else:
                x = 0
            for tau in self.ForwardStage[s].RangePeriodEndItemInv:
                    t = self.ForwardStage[s].TimeObservationStage + tau
                    w.Demands[t] = copy.deepcopy(self.SetOfSAAScenario[t][x])
                    w.Probability *= self.Saascenarioproba[t][x]

        return w

    def CreateAllScenarioFromSAA(self):

        scenarioset=[]

        nrscenar = 1
        for s in range(len(self.StagesSet)):
            if self.ForwardStage[s].TimeObservationStage >= 0:
                nrscenar *= self.NrSAAScenarioInPeriod[self.ForwardStage[s].TimeObservationStage]

        indexstage = [0] * len(self.StagesSet)

        for nrw in range(nrscenar):
            w = Scenario(proabability=1)
            w.Demands = [[] for t in self.Instance.TimeBucketSet]
            for s in range(len(self.StagesSet)):# in self.Instance.TimeBucketSet:

                for tau in self.ForwardStage[s].RangePeriodEndItemInv:
                        t = self.ForwardStage[s].TimeObservationStage + tau
                        w.Demands[t] = copy.deepcopy(self.SetOfSAAScenario[t][indexstage[s]])
                        w.Probability *= self.Saascenarioproba[t][indexstage[s]]

            k=len(self.StagesSet)-1
            stop = False
            while not stop:
                indexstage[k] = indexstage[k] + 1
                if self.ForwardStage[k].TimeObservationStage >= 0 \
                    and indexstage[k] >= self.NrSAAScenarioInPeriod[self.ForwardStage[k].TimeObservationStage]:
                    indexstage[k] = 0
                    k=k-1
                else:
                    stop = True

            scenarioset.append(w)


        return scenarioset

    #This function return the quanity of product to produce at time which has been decided at an earlier stage
    def GetQuantityFixedEarlier(self, product, time, scenario):
        forwardstage = self.ForwardStageWithQuantityDec[time]
        t = forwardstage.GetTimeIndexForQty(product, time)
        if self.UseCorePoint and forwardstage.IsFirstStage():
            result = forwardstage.CorePointQuantityValues[scenario][t][product]
        else:
            result = forwardstage.QuantityValues[scenario][t][product]

        return result

    # This function return the inventory  of product  at time which has been decided at an earlier stage
    def GetInventoryFixedEarlier(self, product, time, scenario):
        if time == -1:
            result = self.Instance.StartingInventories[product]
        else:
            decisiontime = 0
            if self.Instance.HasExternalDemand[product]:
                forwardstage = self.ForwardStageWithBackOrderDec[time]
            else:
                forwardstage = self.ForwardStageWithCompInvDec[time]

            t = forwardstage.GetTimeIndexForInv(product, time)
            if self.UseCorePoint and forwardstage.IsFirstStage():
                result = forwardstage.CorePointInventoryValue[scenario][t][product]
            else:
                result = forwardstage.InventoryValue[scenario][t][product]

        return result

    # This function return the backordered quantity of product which has been decided at an earlier stage
    def GetBackorderFixedEarlier(self, product, time, scenario):
        if time == -1:
            result = 0
        else:
            forwardstage = self.ForwardStageWithBackOrderDec[time]
            t = forwardstage.GetTimeIndexForBackorder(product, time)
            if self.UseCorePoint and forwardstage.IsFirstStage():
                result = forwardstage.CorePointBackorderValue[scenario][t][product]
            else:
                result = forwardstage.BackorderValue[scenario][t][product]

        return result

    #This function return the value of the setup variable of product to produce at time which has been decided at an earlier stage
    def GetSetupFixedEarlier(self, product, time, scenario):
        if self.UseCorePoint:
            result = self.ForwardStage[0].CorePointProductionValue[scenario][time][product]
        else:
            result = self.ForwardStage[0].ProductionValue[scenario][time][product]
        return result

    #This function return the demand of product at time in scenario
    def GetDemandQuantity(self, product, time, scenario):
        result = 0
        print("Is that called????")
        return result

    #This funciton update the lower bound based on the last forward pass
    def UpdateLowerBound(self):
        result = self.ForwardStage[0].PassCostWithAproxCosttoGo

        if self.CurrentLowerBound > 0 and result - self.CurrentLowerBound <= self.CurrentToleranceForSameLB:
            self.NrIterationWithoutLBImprovment += 1
        else:
            self.NrIterationWithoutLBImprovment = 0
        self.CurrentLowerBound = result

    #This funciton update the upper bound based on the last forward pass
    def UpdateUpperBound(self):
            laststage = len(self.StagesSet) - 1
#            expectedupperbound = sum( self.Stage[ s ].PassCost for s in self.StagesSet )
#            variance = math.pow( np.std(  [ sum( self.Stage[ s ].PartialCostPerScenario[w] for s in self.StagesSet ) for w in self.ScenarioNrSet]  ), 2 )
            expectedupperbound = self.ForwardStage[laststage].PassCost
            variance = sum(math.pow(expectedupperbound-self.ForwardStage[laststage].PartialCostPerScenario[w], 2) for w in self.TrialScenarioNrSet) / self.CurrentNrScenario
            self.CurrentUpperBound = expectedupperbound - 1.96 * math.sqrt(variance / self.CurrentNrScenario)
            self.CurrentExpvalueUpperBound = expectedupperbound
            self.CurrentSafeUpperBound = expectedupperbound + 1.96 * math.sqrt(variance / self.CurrentNrScenario)
            self.VarianceForwardPass = variance

    #This function check if the stopping criterion of the algorithm is met
    def CheckStoppingCriterion(self):

        duration = time.time() - self.StartOfAlsorithm
        timalimiteached = (duration > Constants.AlgorithmTimeLimit)
        optimalitygap = (self.CurrentExpvalueUpperBound - self.CurrentLowerBound)/self.CurrentExpvalueUpperBound

        if Constants.PrintSDDPTrace:
            self.WriteInTraceFile("Iteration: %d, Duration: %d, LB: %r, (exp UB:%r),  Gap: %r,  Fixed Y: %r  \n"
                                  % (self.CurrentIteration, duration, self.CurrentLowerBound,
                                     self.CurrentExpvalueUpperBound,
                                     optimalitygap, self.HasFixedSetup))

        return timalimiteached



        convergencecriterion = Constants.Infinity
        c = Constants.Infinity
        if self.CurrentLowerBound > 0:
            convergencecriterion = float(self.CurrentUpperBound) / float(self.CurrentLowerBound) \
                                   - (1.96 * math.sqrt(float(self.VarianceForwardPass) \
                                                     / float(self.CurrentNrScenario)) \
                                         / float(self.CurrentLowerBound))

            c = (1.96 * math.sqrt(float(self.VarianceForwardPass) / float(self.CurrentNrScenario)) \
                                     / float(self.CurrentLowerBound))

        delta = Constants.Infinity
        if self.CurrentLowerBound > 0:
            delta = 3.92 * math.sqrt(float(self.VarianceForwardPass) / float(self.CurrentNrScenario)) \
                    / float(self.CurrentLowerBound)

        convergencecriterionreached = convergencecriterion <= 1 \
                                      and delta <= Constants.AlgorithmOptimalityTolerence



        optimalitygapreached = (optimalitygap < Constants.AlgorithmOptimalityTolerence)
        iterationlimitreached = (self.CurrentIteration > Constants.SDDPIterationLimit)
        result = ( self.IsIterationWithConvergenceTest and convergencecriterionreached) \
                 or timalimiteached \
                 or iterationlimitreached

        if Constants.SDDPForwardPassInSAATree:
            result = (self.IsIterationWithConvergenceTest and optimalitygapreached) \
                     or timalimiteached \
                     or iterationlimitreached

        if Constants.PrintSDDPTrace:
            if self.IsIterationWithConvergenceTest:
                self.WriteInTraceFile(
                    "Convergence Test, Nr Scenario: %r, Duration: %d, LB: %r, (exp UB:%r), c: %r Gap: %r, conv: %r, delta : %r Fixed Y: %r \n"
                    % (self.SDDPNrScenarioTest, duration, self.CurrentLowerBound, self.CurrentExpvalueUpperBound,
                       c, optimalitygap, convergencecriterion, delta,  self.HasFixedSetup))
            else:
                self.WriteInTraceFile("Iteration: %d, Duration: %d, LB: %r, (exp UB:%r), c: %r Gap: %r, conv: %r, delta : %r Fixed Y: %r  \n"
                                  %(self.CurrentIteration, duration, self.CurrentLowerBound, self.CurrentExpvalueUpperBound,
                                    c, optimalitygap, convergencecriterion, delta,  self.HasFixedSetup))

        if not result and convergencecriterion <= 1 \
            and (self.IsIterationWithConvergenceTest \
                or ((not Constants.SDDPForwardPassInSAATree)
                     and ((self.CurrentIteration - self.LastIterationWithTest) > Constants.SDDPMinimumNrIterationBetweenTest))

                 or (Constants.SDDPForwardPassInSAATree
                         and (self.NrIterationWithoutLBImprovment) > Constants.SDDPNrItNoImproveLBBeforeTest)):

            if self.IsIterationWithConvergenceTest:
                self.SDDPNrScenarioTest += Constants.SDDPIncreaseNrScenarioTest
            self.LastIterationWithTest = self.CurrentIteration

            if Constants.SDDPForwardPassInSAATree and self.IsIterationWithConvergenceTest:
                self.IsIterationWithConvergenceTest = False
            else:
                if Constants.SDDPPerformConvergenceTestDuringRun:
                    self.IsIterationWithConvergenceTest = True
            result = False
            self.GenerateTrialScenarios()
            self.ForwardPass()
            self.ComputeCost()
            self.UpdateUpperBound()
            self.LastExpectedCostComputedOnAllScenario = self.CurrentExpvalueUpperBound
            result = self.CheckStoppingCriterion()
        else:
            self.IsIterationWithConvergenceTest = False
            #self.SDDPNrScenarioTest = Constants.SDDPInitNrScenarioTest

        duration = time.time() - self.StartOfAlsorithm

        return result

    def CheckStoppingRelaxationCriterion(self, phase):
        duration = time.time() - self.StartOfAlsorithm

        optimalitygap = (self.CurrentExpvalueUpperBound - self.CurrentLowerBound)/self.CurrentExpvalueUpperBound
        optimalitygapreached = (optimalitygap < Constants.SDDPGapRelax)
        iterationlimitreached = (self.CurrentIteration > Constants.SDDPNrIterationRelax * phase)
        durationlimitreached = duration > Constants.AlgorithmTimeLimit
        result = iterationlimitreached or durationlimitreached# or optimalitygapreached
        self.WriteInTraceFile("Iteration: %d, Duration: %d, LB: %r, (exp UB:%r),  Gap: %r \n"
                              % (self.CurrentIteration, duration, self.CurrentLowerBound, self.CurrentExpvalueUpperBound,
                              optimalitygap,))
        return result

    #This funciton compute the solution of the scenario given in argument (used after to have run the algorithm, and the cost to go approximation are built)
    def ComputeSolutionForScenario(self, scenario):
        solution = Solution()
        return solution

    def WriteInTraceFile(self, string):
        if Constants.PrintSDDPTrace:
            self.TraceFile = open(self.TraceFileName, "a")
            self.TraceFile.write(string)
            self.TraceFile.close()

    def SolveTwoStageHeuristic(self):


        # if Constants.SDDPForwardPassInSAATree:
        #     subsetofscenario =[]
        #     for w in range(200):
        #         selected = self.CreateRandomScenarioFromSAA()
        #         selected.Probability = 1/200.0
        #         subsetofscenario.append(selected)
        #
        #
        #     treestructure = [1, len(subsetofscenario)] + [1] * (self.Instance.NrTimeBucket - 1) + [0]
        #
        #     scenariotree = ScenarioTree(self.Instance, treestructure, 0,
        #                                 CopyscenariofromYFIX=True,
        #                                 givenscenarioset=subsetofscenario)
        #
        # else:
        treestructur = [1, 200] + [1] * (self.Instance.NrTimeBucket - 1) + [0]
        scenariotree = ScenarioTree(self.Instance, treestructur, 0,
                                        scenariogenerationmethod=Constants.RQMC )


        mipsolver = MIPSolver(self.Instance, Constants.ModelYQFix, scenariotree)

        mipsolver.BuildModel()
        if Constants.Debug:
            print("Start to solve instance %s with Cplex" % self.Instance.InstanceName)

        self.TwoStageSolution = mipsolver.Solve()
        self.HeuristicSetupValue = [[self.TwoStageSolution.Production[0][t][p]
                                     for p in self.Instance.ProductSet]
                                    for t in self.Instance.TimeBucketSet]



        self.CurrentBestSetups = self.HeuristicSetupValue


    #This function runs the SDDP algorithm
    def Run(self):
        if Constants.PrintSDDPTrace:
            self.TraceFile = open(self.TraceFileName, "w")

            self.TraceFile.write("Start the SDDP algorithm \n")
            self.TraceFile.write("Use Papadakos method to generate strong cuts: %r \n"%Constants.GenerateStrongCut)
            self.TraceFile.write("Generate  cuts with linear relaxation: %r \n"%Constants.SolveRelaxationFirst)
            self.TraceFile.write("Generate  cuts with two-stage solution: %r \n" % Constants.SolveRelaxationFirst)
            self.TraceFile.write("Use valid inequalities: %r \n"%Constants.SDDPUseValidInequalities)
            self.TraceFile.write("Run SDDP in a single tree: %r \n"%Constants.SDDPRunSigleTree)
            self.TraceFile.write("SDDP setting: %r \n"%self.TestIdentifier.SDDPSetting )
            self.TraceFile.close()
        self.StartOfAlsorithm = time.time()
       # print("Attention SDDP solve average")

        self.GenerateSAAScenarios2()
        if Constants.Debug:
            print("********************Scenarios SAA*********************")
            for t in self.Instance.TimeBucketSet:
                for w in self.SAAScenarioNrSetInPeriod[t]:
                    print("SAA demand at stage %r in scenario %r: %r" % (t, w, self.SetOfSAAScenario[t][w]))

        #if Constants.SDDPGenerateCutWith2Stage:




        if self.TestIdentifier.Model == Constants.ModelHeuristicYFix or self.TestIdentifier.Method == Constants.Hybrid or self.TestIdentifier.SDDPSetting == Constants.JustYFix:
            Constants.SDDPGenerateCutWith2Stage = False
            Constants.SolveRelaxationFirst = False
            Constants.SDDPRunSigleTree = False
        else:
            self.SolveTwoStageHeuristic()

        if Constants.Debug:
            print("Heuristic Setup: %r" % self.HeuristicSetupValue)
        createpreliminarycuts = Constants.SolveRelaxationFirst or Constants.SDDPGenerateCutWith2Stage
        phase = 1
        if not Constants.SolveRelaxationFirst:
            phase = 2
        ExitLoop = not createpreliminarycuts and Constants.SDDPRunSigleTree
        Stop = False
        while (not Stop or createpreliminarycuts) and not ExitLoop:

            if createpreliminarycuts and (Stop or self.CheckStoppingRelaxationCriterion(phase)):
                phase += 1
                if phase < 3:
                    self.ForwardStage[0].ChangeSetupToValueOfTwoStage()
                    self.WriteInTraceFile("Change stage 1 problem to heuristic solution \n")
                    self.CurrentUpperBound = Constants.Infinity
                    self.LastExpectedCostComputedOnAllScenario = Constants.Infinity
                    self.CurrentLowerBound = 0
                    self.NrIterationWithoutLBImprovment = 0

                else:

                    #elif round == 3:
                    #    print("round3")
                    ExitLoop = Constants.SDDPRunSigleTree
                    #    print(ExitLoop)
                    createpreliminarycuts = False
                    self.CurrentUpperBound = Constants.Infinity
                    #self.LastExpectedCostComputedOnAllScenario = Constants.Infinity
                    self.CurrentLowerBound = 0
                    self.NrIterationWithoutLBImprovment = 0
                    #Make a convergence test after adding cuts of the two stage

                    if not ExitLoop:
                        self.ForwardStage[0].ChangeSetupToBinary()
                        self.WriteInTraceFile("Change stage 1 problem to integer \n")


            #print("Attention uncomment")
            self.IsIterationWithConvergenceTest = False

            self.GenerateTrialScenarios()

            if Constants.Debug:
                print("********************Scenarios Trial*********************")
                for s in self.CurrentSetOfTrialScenarios:
                    print("Demands: %r" % s.Demands)
                if Constants.SDDPForwardPassInSAATree:
                    print("********************Scenarios SAA*********************")
                    for t in self.Instance.TimeBucketSet:
                        for w in self.SAAScenarioNrSetInPeriod[t]:
                            print("SAA demand at stage %r in scenario %r: %r" % (t, w, self.SetOfSAAScenario[t][w]))

                print("********************Scenarios EVPI*********************")
                if Constants.SDDPUseEVPI and self.ForwardStage[0].EVPIScenarioSet is not None:
                    for s in self.ForwardStage[0].EVPIScenarioSet:
                        print("Demands: %r" % s.Demands)
                print("****************************************************")
            if Constants.SDDPModifyBackwardScenarioAtEachIteration:
                self.GenerateSAAScenarios2()
            self.ForwardPass()
            self.ComputeCost()
            self.UpdateLowerBound()
            self.UpdateUpperBound()
            self.BackwardPass()
            self.CurrentIteration = self.CurrentIteration + 1



            if not createpreliminarycuts:# and (not self.HasFixedSetup or self.TestIdentifier.Model == Constants.ModelHeuristicYFix):
                Stop = self.CheckStoppingCriterion()
            #if self.HasFixedSetup and not self.TestIdentifier.Model == Constants.ModelHeuristicYFix:
            else:
                self.WriteInTraceFile("Iteration With Fixed Setup  LB: % r, (exp UB: % r) \n"% (self.CurrentLowerBound, self.CurrentExpvalueUpperBound))
            duration = time.time() - self.StartOfAlsorithm

           # if self.CurrentForwardSampleSize < 10 and duration >= Constants.SDDPDurationBeforeIncreaseForwardSample:
           #     self.CurrentForwardSampleSize = 10
           #     self.WriteInTraceFile("set number of scenario in forward to 10 \n")
           # if self.CurrentForwardSampleSize < 50 and duration >= 3*Constants.SDDPDurationBeforeIncreaseForwardSample:
           #     self.CurrentForwardSampleSize = 50
           #     self.WriteInTraceFile("set number of scenario in forward to 50 \n")

            newsetup = [[round(self.ForwardStage[0].ProductionValue[0][t][p], 0)
                         for p in self.Instance.ProductSet]
                        for t in self.Instance.TimeBucketSet]


          #  if not createpreliminarycuts and self.CurrentSetups == newsetup and Constants.SDDPFixSetupStrategy and not self.HasFixedSetup:
          #      self.HeuristicSetupValue = newsetup
          #      self.ForwardStage[0].ChangeSetupToValueOfTwoStage()
          #      self.IterationSetupFixed = self.CurrentIteration
          #      self.HasFixedSetup = True

          #  if self.HasFixedSetup and self.IterationSetupFixed < self.CurrentIteration - 10:
          #      self.ForwardStage[0].ChangeSetupToBinary()
          #      self.HasFixedSetup = False

         #   self.CurrentSetups = newsetup
            #if self.CurrentForwardSampleSize < 100 and duration >= 3*Constants.SDDPDurationBeforeIncreaseForwardSample:
            #    self.CurrentForwardSampleSize = 100
            #    self.WriteInTraceFile("set number of scenario in forward to 100 \n")

            #if self.CurrentIteration % 100 < 5:
            #    solution = self.CreateSolutionOfScenario(0)
            #    solution.PrintToExcel("solution_at_iteration_%r"%self.CurrentIteration)

        if Constants.SDDPRunSigleTree:
            self.RunSingleTreeSDDP()

        self.SDDPNrScenarioTest = 1000
        random.seed = 9876

        self.ComputeUpperBound()

        self.RecordSolveInfo()
        if Constants.PrintSDDPTrace:
            self.WriteInTraceFile("End of the SDDP algorithm \n ")




    def ComputeUpperBound(self):
        self.IsIterationWithConvergenceTest = True
        self.GenerateTrialScenarios()
        self.ForwardPass()
        self.ComputeCost()
        self.UpdateUpperBound()
        self.LastExpectedCostComputedOnAllScenario = self.CurrentExpvalueUpperBound
        self.WriteInTraceFile(
            "Convergence Test, Nr Scenario: %r, LB: %r, (exp UB:%r - safe UB = %r),  Fixed Y: %r \n"
            % (self.SDDPNrScenarioTest,  self.CurrentLowerBound, self.CurrentExpvalueUpperBound, self.CurrentSafeUpperBound,  self.HasFixedSetup))

    # This function runs the SDDP algorithm
    def RunSingleTreeSDDP(self):

        if not Constants.SolveRelaxationFirst:
            #run forward pass to create the MIPS
            Constants.SolveRelaxationFirst = True
            self.GenerateTrialScenarios()
            self.ForwardPass()
            Constants.SolveRelaxationFirst = False

        self.IsIterationWithConvergenceTest = True
        self.GenerateTrialScenarios()
        self.ForwardPass()
        self.CheckStoppingCriterion()
        self.BestUpperBound = self.LastExpectedCostComputedOnAllScenario

        #Make a copy to be able to solve the first stage with contiunous variable in the call backs
        self.CopyFirstStage = SDDPStage(owner=self, decisionstage=0, fixedccenarioset=[0], isforward=True)
        self.CopyFirstStage.SetNrTrialScenario(len(self.CurrentSetOfTrialScenarios))
        for cut in self.ForwardStage[0].SDDPCuts:
            cut.ForwardStage = None
            cut.BackwarStage = None


        self.CopyFirstStage.SDDPCuts = copy.deepcopy(self.ForwardStage[0].SDDPCuts)

        for cut in self.ForwardStage[0].SDDPCuts:
            cut.ForwardStage = self.ForwardStage[0]
            cut.BackwarStage = self.ForwardStage[0]

        for cut in self.CopyFirstStage.SDDPCuts:
            cut.ForwardStage = self.CopyFirstStage
            cut.BackwarStage = self.CopyFirstStage


        self.CopyFirstStage.ComputeVariablePeriods()
        self.CopyFirstStage.TimeDecisionStage = 0
        self.CopyFirstStage.FixedScenarioSet = [0]
        self.CopyFirstStage.FixedScenarioPobability = [1]
        self.CopyFirstStage.ComputeVariableIndices()
        self.CopyFirstStage.ComputeVariablePeriodsInLargeMIP()
        self.CopyFirstStage.DefineMIP()

        for cut in self.CopyFirstStage.SDDPCuts:
            cut.ForwardStage = self.CopyFirstStage
            cut.AddCut()

        self.CopyFirstStage.ChangeSetupToBinary()

        vars = []
        righthandside = []
        # Setup equal to the given ones

        for p in self.Instance.ProductSet:
            for t in self.Instance.TimeBucketSet:
                    vars = vars + [self.CopyFirstStage.GetIndexProductionVariable(p,t)]
                    righthandside = righthandside + [round(self.HeuristicSetupValue[t][p], 0)]
        self.CopyFirstStage.Cplex.MIP_starts.add(cplex.SparsePair(vars, righthandside),
                                                 self.CopyFirstStage.Cplex.MIP_starts.effort_level.solve_fixed)


        #print("ATTENTION USE CORRIDOR")
        #nrsetups = sum(round(self.HeuristicSetupValue[t][p], 0)
        #                for p in self.Instance.ProductSet
        #                for t in self.Instance.TimeBucketSet)
        #self.CopyFirstStage.CreateCoridorConstraints(nrsetups)

        model_lazy = self.CopyFirstStage.Cplex.register_callback(SDDPCallBack)
        model_lazy.SDDPOwner = self
        model_lazy.Model = self.CopyFirstStage


       # model_usercut = self.CopyFirstStage.Cplex.register_callback(SDDPUserCutCallBack)
       # model_usercut.SDDPOwner = self
       # model_usercut.Model = self.CopyFirstStage
        if Constants.Debug:
            self.CopyFirstStage.Cplex.write("./Temp/MainModel.lp")
        cplexlogfilename = "./Temp/CPLEXLog_%s_%s.txt"%(self.Instance.InstanceName, self.TestIdentifier.MIPSetting)
        self.CopyFirstStage.Cplex.set_log_stream(cplexlogfilename)
        self.CopyFirstStage.Cplex.set_results_stream(cplexlogfilename)
        self.CopyFirstStage.Cplex.set_warning_stream(cplexlogfilename)
        self.CopyFirstStage.Cplex.set_error_stream(cplexlogfilename)
        self.CopyFirstStage.Cplex.parameters.mip.interval.set(1)
        if Constants.Debug:
            print("Start To solve the main tree")
        self.CopyFirstStage.Cplex.parameters.timelimit.set(Constants.AlgorithmTimeLimit)
        self.CopyFirstStage.Cplex.parameters.mip.limits.treememory.set(700000000.0)
        self.CopyFirstStage.Cplex.parameters.threads.set(1)
        self.CopyFirstStage.Cplex.solve()
        self.WriteInTraceFile("End Solve in one tree cost: %r " % self.CopyFirstStage.Cplex.solution.get_objective_value())

        self.SingleTreeCplexGap = self.CopyFirstStage.Cplex.solution.MIP.get_mip_relative_gap()
        #for cut in self.CopyFirstStage.SDDPCuts:
        #    self.ForwardStage[0].SDDPCuts.append(cut)
        #    cut.ForwardStage = self.ForwardStage[0]
        #    cut.AddCut()
        self.IsIterationWithConvergenceTest = True
        self.GenerateTrialScenarios()
        self.HeuristicSetupValue = self.CurrentBestSetups
        self.ForwardStage[0].ChangeSetupToValueOfTwoStage()
        self.ForwardPass()
        self.ComputeCost()
        #self.ForwardStage[0] = self.CopyFirstStage
        self.CurrentLowerBound = self.ForwardStage[0].Cplex.solution.get_objective_value()
        self.NrIterationWithoutLBImprovment = 0
        self.UpdateUpperBound()
        self.LastExpectedCostComputedOnAllScenario = self.CurrentExpvalueUpperBound
        self.ForwardStage[0].SaveSolutionFromSol(self.ForwardStage[0].Cplex.solution)
        #self.ForwardStage[0].CopyDecisionOfScenario0ToAllScenario()
        #self.ForwardStage[0].Cplex.unregister_callback(SDDPCallBack)

        #self.LinkStages()
        #Make a last forward pass to find the optimal insample solution







    def ComputeCost(self):
        for stage in self.ForwardStage:
            stage.ComputePassCost()
            #stage.PassCost = sum( stage.StageCostPerScenario[w] for w in self.ScenarioNrSet  ) / len(self.ScenarioNrSet)

    def SetCurrentBigM(self):
       self.CurrentBigM =[MIPSolver.GetBigMValue(self.Instance, self.CompleteSetOfSAAScenario, p) for p in self.Instance.ProductSet]

    def GetFileName(self):
        result ="./Solutions/SDDP_%s_%s_%s_Solution.sddp"%(self.Instance.InstanceName, self.StartingSeed, self.CurrentNrScenario)
        return result

#    def SaveInFile(self):
#        filename = self.GetFileName()
#        with open(filename , "w+") as fp:
#            pickle.dump(self.Stage[0].SDDPCuts[0], fp)

#    def ReadFromFile(self):
#        filename = self.GetFileName()
#        with open(filename , "rb") as fp:
#            pickle.load(self, fp)

    def RecordSolveInfo(self):
        self.SolveInfo = [self.Instance.InstanceName,
                          "SDDP",
                          self.LastExpectedCostComputedOnAllScenario,
                          self.CurrentLowerBound,
                          self.CurrentIteration,
                          time.time() - self.StartOfAlsorithm,
                          0,
                          0,
                          #sol.progress.get_num_iterations(),
                          #sol.progress.get_num_nodes_processed(),
                          0,
                          0,
                          0,
                          0,
                          0,
                          0,
                          0,
                          self.Instance.NrLevel,
                          self.Instance.NrProduct,
                          self.Instance.NrTimeBucket,
                          0,
                          self.CurrentNrScenario,
                          self.Instance.MaxLeadTime,
                          0]

    def CreateSolutionAtFirstStage(self):
           # Get the setup quantitities associated with the solultion
            solproduction = [[[self.GetSetupFixedEarlier(p, t, 0) for p in self.Instance.ProductSet]
                          for t in self.Instance.TimeBucketSet]]

            solquantity = [[[self.GetQuantityFixedEarlier(p, 0, 0) for p in self.Instance.ProductSet]]]

            solinventory = [[[self.GetInventoryFixedEarlier(p, 0, 0)
                            if not self.Instance.HasExternalDemand[p]
                            else -1
                            for p in self.Instance.ProductSet]]]

            solbackorder = [[[-1 for p in self.Instance.ProductSet]]]

            solconsumption = [[[[-1 for p in self.Instance.ProductSet] for p in self.Instance.ProductSet]]]
            k=-1
            for c in self.Instance.ConsumptionSet:
                 k += 1
                 solconsumption[0][0][c[0]][c[1]] = self.ForwardStage[0].ConsumptionValues[0][0][k]

            emptyscenariotree = ScenarioTree(instance=self.Instance,
                                             branchperlevel=[0,0,0,0,0],
                                             seed=self.TestIdentifier.ScenarioSeed)# instance = None, branchperlevel = [], seed = -1, mipsolver = None, evaluationscenario = False, averagescenariotree = False,  givenfirstperiod = [], scenariogenerationmethod = "MC", generateasYQfix = False, model = "YFix", CopyscenariofromYFIX=False ):


            solution = Solution(self.Instance, solquantity, solproduction, solinventory, solbackorder, solconsumption,
                                [0],  emptyscenariotree, partialsolution=True)

            solution.IsSDDPSolution = True
            solution.FixedQuantity = [[-1 for p in self.Instance.ProductSet] for t in self.Instance.TimeBucketSet]


            solution.SDDPLB = self.CurrentLowerBound
            solution.SDDPExpUB = self.LastExpectedCostComputedOnAllScenario
            solution.SDDPNrIteration = self.CurrentIteration

            solution.SDDPTimeBackward = self.TimeBackward
            solution.SDDPTimeForwardNoTest = self.TimeForwardNonTest
            solution.SDDPTimeForwardTest = self.TimeForwardTest

            solution.CplexGap = self.SingleTreeCplexGap

            return solution


    def CreateSolutionOfScenario(self, scenario):
           # Get the setup quantitities associated with the solultion
            solproduction = [[[self.GetSetupFixedEarlier(p, t, scenario) for p in self.Instance.ProductSet]
                          for t in self.Instance.TimeBucketSet]]

            solquantity = [[[self.GetQuantityFixedEarlier(p, t, scenario) for p in self.Instance.ProductSet] for t in self.Instance.TimeBucketSet]]

            solinventory = [[[self.GetInventoryFixedEarlier(p, t, scenario)

                            for p in self.Instance.ProductSet]
                             for t in self.Instance.TimeBucketSet]]

            solbackorder = [[[self.GetBackorderFixedEarlier(p, t, scenario)

                            for p in self.Instance.ProductSet]
                             for t in self.Instance.TimeBucketSet]]

            solconsumption = [[[[-1 for p in self.Instance.ProductSet] for p in self.Instance.ProductSet]for t in self.Instance.TimeBucketSet]]

            for t in self.Instance.TimeBucketSet:


                k = -1
                for c in self.Instance.ConsumptionSet:
                     k += 1
                     forwardstage = self.ForwardStageWithQuantityDec[t]
                     time = forwardstage.GetTimeIndexForQty(c[1], t)
                     solconsumption[0][t][c[0]][c[1]] = forwardstage.ConsumptionValues[scenario][time][k]

            emptyscenariotree = ScenarioTree(instance=self.Instance,
                                             branchperlevel=[0, 0, 0, 0, 0],
                                             seed=self.TestIdentifier.ScenarioSeed)# instance = None, branchperlevel = [], seed = -1, mipsolver = None, evaluationscenario = False, averagescenariotree = False,  givenfirstperiod = [], scenariogenerationmethod = "MC", generateasYQfix = False, model = "YFix", CopyscenariofromYFIX=False ):


            solution = Solution(self.Instance, solquantity, solproduction, solinventory, solbackorder, solconsumption,
                                [self.CurrentSetOfTrialScenarios[0]], emptyscenariotree, partialsolution=False)

            solution.IsSDDPSolution = False
            solution.FixedQuantity = [[-1 for p in self.Instance.ProductSet] for t in self.Instance.TimeBucketSet]

            return solution

    def CreateSolutionOfAllInSampleScenario(self):

        completescenarioset = range(len(self.CompleteSetOfSAAScenario))
        # Get the setup quantitities associated with the solultion
        solproduction = [[[self.GetSetupFixedEarlier(p, t, scenario) for p in self.Instance.ProductSet]
                          for t in self.Instance.TimeBucketSet] for scenario in completescenarioset]

        solquantity = [[[self.GetQuantityFixedEarlier(p, t, scenario) for p in self.Instance.ProductSet] for t in
                        self.Instance.TimeBucketSet] for scenario in completescenarioset]

        solinventory = [[[self.GetInventoryFixedEarlier(p, t, scenario)

                          for p in self.Instance.ProductSet]
                         for t in self.Instance.TimeBucketSet] for scenario in completescenarioset]

        solbackorder = [[[self.GetBackorderFixedEarlier(p, t, scenario)

                          for p in self.Instance.ProductWithExternalDemand]
                         for t in self.Instance.TimeBucketSet] for scenario in completescenarioset]

        solconsumption = [[[[-1 for p in self.Instance.ProductSet] for q in self.Instance.ProductSet] for t in
                           self.Instance.TimeBucketSet] for scenario in completescenarioset]
        for scenario in completescenarioset:
            for t in self.Instance.TimeBucketSet:
                k = -1
                for c in self.Instance.ConsumptionSet:
                    k += 1

                    stage = self.ForwardStageWithQuantityDec[t]
                    tindex = stage.GetTimeIndexForQty(c[0], t)
                    solconsumption[scenario][t][c[0]][c[1]] = stage.ConsumptionValues[scenario][tindex][k]

        #emptyscenariotree = ScenarioTree(instance=self.Instance,
        #                                 branchperlevel=[0, 0, 0, 0, 0],
        #                                 seed=self.TestIdentifier.ScenarioSeed)  # instance = None, branchperlevel = [], seed = -1, mipsolver = None, evaluationscenario = False, averagescenariotree = False,  givenfirstperiod = [], scenariogenerationmethod = "MC", generateasYQfix = False, model = "YFix", CopyscenariofromYFIX=False ):

        solution = Solution(self.Instance, solquantity, solproduction, solinventory, solbackorder, solconsumption,
                            self.CompleteSetOfSAAScenario, self.SAAScenarioTree, partialsolution=False)

        solution.IsSDDPSolution = False
        solution.FixedQuantity = [[-1 for p in self.Instance.ProductSet] for t in self.Instance.TimeBucketSet]

        solution.SDDPLB = self.CurrentLowerBound
        solution.SDDPExpUB = self.LastExpectedCostComputedOnAllScenario
        solution.SDDPNrIteration = self.CurrentIteration
        solution.SDDPTimeBackward = self.TimeBackward
        solution.SDDPTimeForwardNoTest = self.TimeForwardNonTest
        solution.SDDPTimeForwardTest = self.TimeForwardTest

        solution.CplexGap = self.SingleTreeCplexGap

        return solution

    def GetSaveFileName(self):
        if Constants.PrintSolutionFileInTMP:
            result = "/tmp/thesim/Solutions/SDDP_%r.pkl" % self.TestIdentifier.GetAsStringList()
        else:
            result = "./Solutions/SDDP_%r.pkl" % self.TestIdentifier.GetAsStringList()
        return result
    def SaveSolver(self):
        cuts = [[] for _ in self.StagesSet]
        for t in self.StagesSet:
            for cut in self.ForwardStage[t].SDDPCuts:
                cut.ForwardStage = None
                cut.BackwarStage = None
                cuts[t].append(cut)

        filename = self.GetSaveFileName()
        with open(filename, 'wb') as output:
            pickle.dump(cuts, output)

        for t in self.StagesSet:
            for cut in cuts[t]:
                cut.ForwardStage = self.ForwardStage[t]
                cut.BackwarStage = self.BackwardStage[t]

    def LoadCuts(self):
        filename = self.GetSaveFileName()

        with open(filename, 'rb') as input:
            cuts = pickle.load(input)

        for t in self.StagesSet:
            for cut in cuts[t]:
                cut.ForwardStage = self.ForwardStage[t]
                self.ForwardStage[t].SDDPCuts.append(cut)
                cut.BackwarStage = self.BackwardStage[t]
                self.BackwardStage[t].SDDPCuts.append(cut)
