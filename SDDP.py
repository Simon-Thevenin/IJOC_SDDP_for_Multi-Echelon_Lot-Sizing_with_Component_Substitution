from __future__ import absolute_import, division, print_function

from Constants import Constants
from Solution import Solution
from SDDPStage import SDDPStage
from SDDPLastStage import SDDPLastStage
from ScenarioTree import ScenarioTree
from MIPSolver import MIPSolver
import numpy as np
import math
import time
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
        for stage in self.Stage:
            stage.PreviousSDDPStage = previousstage
            if not previousstage is None:
                previousstage.NextSDDPStage = stage
            previousstage = stage

    def __init__(self, instance, testidentifier):
        self.Instance = instance
        self.TestIdentifier = testidentifier
        nrstage = self.Instance.NrTimeBucket# - self.Instance.NrTimeBucketWithoutUncertainty
        self.StagesSet = range(nrstage + 1)
        self.CurrentIteration = 0
        self.CurrentLowerBound = 0
        self.CurrentUpperBound = Constants.Infinity
        self.StartOfAlsorithm = time.time()
        self.CurrentSetOfScenarios = []
        self.ScenarioNrSet = []
        self.CurrentScenarioSeed = int(self.TestIdentifier.ScenarioSeed)
        self.StartingSeed = self.TestIdentifier.ScenarioSeed
        self.Stage = [SDDPStage(owner=self, decisionstage=t) for t in range(nrstage)] \
                        + [SDDPLastStage(owner=self, decisionstage=nrstage)]
        self.LinkStages()
        self.CurrentNrScenario = int(self.TestIdentifier.NrScenario)
        self.CurrentBigM = []
        self.ScenarioGenerationMethod = self.TestIdentifier.ScenarioSampling
        self.CurrentExpvalueUpperBound = Constants.Infinity
        self.EvaluationMode = False
        self.UseCorePoint = False
        self.GenerateStrongCut = Constants.GenerateStrongCut
        self.TraceFile = None



    #This function make the forward pass of SDDP
    def ForwardPass(self):
        if Constants.Debug:
            print("Start forward pass")
        self.SetCurrentBigM()
        for t in self.StagesSet:
            #Run the forward pass at each stage t
            self.Stage[t].RunForwardPassMIP()

            #try to use core point method, remove if it does not work
            #if self.Stage[t].IsFirstStage()
            if self.GenerateStrongCut:
                self.Stage[t].UpdateCorePoint()


    #This function make the backward pass of SDDP
    def BackwardPass(self):
        if Constants.Debug:
            print("Start Backward pass")

        self.UseCorePoint = self.GenerateStrongCut

        #rerun last stage to get dual with stron cuts:
        if Constants.GenerateStrongCut:
            self.Stage[len(self.StagesSet)-1].RunForwardPassMIP()

        for t in reversed(range(1, len(self.StagesSet) -1)):
            #Build or update the MIP of stage t
            self.Stage[t].GernerateCut()

        self.UseCorePoint = False


    #This function generates the scenarios for the current iteration of the algorithm
    def GenerateScenarios(self, nrscenario, average = False):
        if Constants.Debug:
            print("Start generation of new scenarios")

        #Generate a scenario tree
        treestructure = [1, nrscenario] + [1] * (self.Instance.NrTimeBucket - 1) + [0]
        scenariotree = ScenarioTree(self.Instance, treestructure, self.CurrentScenarioSeed,
                                    scenariogenerationmethod=self.ScenarioGenerationMethod,
                                    generateasYQfix=False, averagescenariotree=True)

        #Get the set of scenarios
        self.CurrentSetOfScenarios = scenariotree.GetAllScenarios(computeindex=False)
        self.ScenarioNrSet = range(len(self.CurrentSetOfScenarios))

        #Modify the number of scenario at each stage
        for stage in self.StagesSet:
            self.Stage[stage].SetNrScenario(len(self.CurrentSetOfScenarios))

        self.CurrentScenarioSeed = self.CurrentScenarioSeed + 1


    #This function return the quanity of product to produce at time which has been decided at an earlier stage
    def GetQuantityFixedEarlier(self, product, time, scenario):
        if self.UseCorePoint:
            result = self.Stage[time].CorePointQuantityValues[scenario][product]
        else:
            result = self.Stage[time].QuantityValues[scenario][product]

        return result

    # This function return the inventory  of product  at time which has been decided at an earlier stage
    def GetInventoryFixedEarlier(self, product, time, scenario):
        if time == -1:
            result = self.Instance.StartingInventories[product]
        else:
            decisiontime = 0
            if self.Instance.HasExternalDemand[product]:
                decisiontime = time + 1
            else:
                decisiontime = time

            if self.UseCorePoint:
                result = self.Stage[decisiontime].CorePointInventoryValue[scenario][product]
            else:
                result = self.Stage[decisiontime].InventoryValue[scenario][product]

        return result

    # This function return the backordered quantity of product which has been decided at an earlier stage
    def GetBackorderFixedEarlier(self, product, time, scenario):
        if time == -1:
            result = 0
        else:
            if self.UseCorePoint:
                result = self.Stage[time + 1].CorePointBackorderValue[scenario][self.Instance.ProductWithExternalDemandIndex[product]]
            else:
                result = self.Stage[time + 1].BackorderValue[scenario][self.Instance.ProductWithExternalDemandIndex[product]]

        return result

    #This function return the value of the setup variable of product to produce at time which has been decided at an earlier stage
    def GetSetupFixedEarlier(self, product, time, scenario):
        if self.UseCorePoint:
            result = self.Stage[0].CorePointProductionValue[scenario][time][product]
        else:
            result = self.Stage[0].ProductionValue[scenario][time][product]
        return result

    #This function return the demand of product at time in scenario
    def GetDemandQuantity(self, product, time, scenario):
        result = 0
        return result

    #This funciton update the lower bound based on the last forward pass
    def UpdateLowerBound(self):
        result = self.Stage[0].PassCostWithAproxCosttoGo
        self.CurrentLowerBound = result

    #This funciton update the upper bound based on the last forward pass
    def UpdateUpperBound(self):
            laststage = len(self.StagesSet) - 1
#            expectedupperbound = sum( self.Stage[ s ].PassCost for s in self.StagesSet )
#            variance = math.pow( np.std(  [ sum( self.Stage[ s ].PartialCostPerScenario[w] for s in self.StagesSet ) for w in self.ScenarioNrSet]  ), 2 )
            expectedupperbound = self.Stage[laststage].PassCost
            variance = math.pow(np.std(self.Stage[laststage].PartialCostPerScenario), 2)
            self.CurrentUpperBound = expectedupperbound + 3.67 * math.sqrt(variance / self.CurrentNrScenario)
            self.CurrentExpvalueUpperBound = expectedupperbound

    #This function check if the stopping criterion of the algorithm is met
    def CheckStoppingCriterion(self):
        duration = time.time() - self.StartOfAlsorithm
        timalimiteached = (duration > Constants.AlgorithmTimeLimit)
        optimalitygap = ( self.CurrentUpperBound - self.CurrentLowerBound) / self.CurrentUpperBound
        optimalitygapreached = (optimalitygap < Constants.AlgorithmOptimalityTolerence)
        iterationlimitreached = (self.CurrentIteration > Constants.SDDPIterationLimit)
        result = optimalitygapreached or timalimiteached or iterationlimitreached
        if Constants.PrintSDDPTrace:
            self.TraceFile.write("Iteration: %d, Duration: %d, LB: %r, UB: %r (exp:%r), Gap: %r \n" %(self.CurrentIteration, duration, self.CurrentLowerBound, self.CurrentUpperBound,  self.CurrentExpvalueUpperBound, optimalitygap))
        return result

    #This funciton compute the solution of the scenario given in argument (used after to have run the algorithm, and the cost to go approximation are built)
    def ComputeSolutionForScenario(self, scenario):
        solution = Solution()
        return solution

    #This function runs the SDDP algorithm
    def Run(self):
        if Constants.PrintSDDPTrace:
            self.TraceFile = open("./Temp/trace.txt", "w")

            self.TraceFile.write("Start the SDDP algorithm \n")
            self.TraceFile.write("Use Papadakos method to generate strong cuts: %r \n"%Constants.GenerateStrongCut)
            self.TraceFile.write("Generate a 1000 cuts with linear relaxation: %r \n"%Constants.GenerateStrongCut)

        self.StartOfAlsorithm = time.time()
        self.GenerateScenarios(self.CurrentNrScenario, average=True)
        while not self.CheckStoppingCriterion():

            if Constants.SolveRelaxationFirst and self.CurrentIteration == 1000:
                self.Stage[0].ChangeSetupToBinary()
                self.TraceFile.write("Change stage 1 problem to integer \n")

            #self.GenerateScenarios(self.CurrentNrScenario)
            if Constants.Debug:
                print("********************Scenarios*********************")
                for s in self.CurrentSetOfScenarios:
                    print("Demands: %r" % s.Demands)
                print("****************************************************")

            self.ForwardPass()
            self.ComputeCost()
            self.UpdateLowerBound()
            self.UpdateUpperBound()
            self.BackwardPass()
            self.CurrentIteration = self.CurrentIteration + 1
            #if self.CurrentIteration % 50 == 0:
            #    self.CurrentNrScenario =  self.CurrentNrScenario  * 10

            if self.CurrentIteration % 100 < 5:
                solution = self.CreateSolutionOfScenario(0)
                solution.PrintToExcel("solution_at_iteration_%r"%self.CurrentIteration)

        self.RecordSolveInfo()
        if Constants.PrintSDDPTrace:
            self.TraceFile.write("End of the SDDP algorithm \n")
        self.TraceFile.close()

    def ComputeCost(self):
        for stage in self.Stage:
            stage.ComputePassCost()
            #stage.PassCost = sum( stage.StageCostPerScenario[w] for w in self.ScenarioNrSet  ) / len(self.ScenarioNrSet)

    def SetCurrentBigM(self):
       self.CurrentBigM =[MIPSolver.GetBigMValue(self.Instance, self.CurrentSetOfScenarios, p) for p in self.Instance.ProductSet]

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
                          self.CurrentUpperBound,
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
                 solconsumption[0][0][c[0]][c[1]] = self.Stage[0].ConsumptionValues[0][k]

            emptyscenariotree = ScenarioTree(instance=self.Instance,
                                             branchperlevel=[0,0,0,0,0],
                                             seed=self.TestIdentifier.ScenarioSeed)# instance = None, branchperlevel = [], seed = -1, mipsolver = None, evaluationscenario = False, averagescenariotree = False,  givenfirstperiod = [], scenariogenerationmethod = "MC", generateasYQfix = False, model = "YFix", CopyscenariofromYFIX=False ):


            solution = Solution(self.Instance, solquantity, solproduction, solinventory, solbackorder, solconsumption,
                                [0],  emptyscenariotree, partialsolution=True)

            solution.IsSDDPSolution = True
            solution.FixedQuantity = [[-1 for p in self.Instance.ProductSet] for t in self.Instance.TimeBucketSet]

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
                     solconsumption[scenario][t][c[0]][c[1]] = self.Stage[t].ConsumptionValues[scenario][k]

            emptyscenariotree = ScenarioTree(instance=self.Instance,
                                             branchperlevel=[0,0,0,0,0],
                                             seed=self.TestIdentifier.ScenarioSeed)# instance = None, branchperlevel = [], seed = -1, mipsolver = None, evaluationscenario = False, averagescenariotree = False,  givenfirstperiod = [], scenariogenerationmethod = "MC", generateasYQfix = False, model = "YFix", CopyscenariofromYFIX=False ):


            solution = Solution(self.Instance, solquantity, solproduction, solinventory, solbackorder, solconsumption,
                                [self.CurrentSetOfScenarios[0]],  emptyscenariotree, partialsolution=False)

            solution.IsSDDPSolution = False
            solution.FixedQuantity = [[-1 for p in self.Instance.ProductSet] for t in self.Instance.TimeBucketSet]

            return solution