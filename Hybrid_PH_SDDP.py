# This class contains the attributes and methods allowing to define the progressive hedging algorithm.
from ScenarioTree import ScenarioTree
from Constants import Constants
from MIPSolver import MIPSolver
from SDDP import SDDP
from ProgressiveHedging import ProgressiveHedging

from Solution import Solution

import copy
import time
import math

class Hybrid_PH_SDDP(object):

    def __init__(self, instance, testidentifier, treestructure, solver):
        self.Instance = instance
        self.TestIdentifier = testidentifier
        self.TreeStructure = treestructure

 #       self.TraceFileName = "./Temp/SDDPPHtrace_%s.txt" % (self.TestIdentifier.GetAsString())
#
        self.Solver = solver


        OldNrScenar =self.TestIdentifier.NrScenario
        self.TestIdentifier.NrScenario = "6400b"
        PHTreestructure = solver.GetTreeStructure()
        # treestructure = [1, 200] + [1] * (self.Instance.NrTimeBucket - 1) + [0]
        # self.TestIdentifier.Model = Constants.ModelYQFix
        # chosengeneration = self.TestIdentifier.ScenarioSampling
        # self.ScenarioGeneration = "RQMC"
        # solution, mipsolver = self.Solver.MRP(treestructure, False, recordsolveinfo=True)
        # givensetup = [[solution.Production[0][t][p] for p in self.Instance.ProductSet]
        #                                       for t in self.Instance.TimeBucketSet]
        self.ProgressiveHedging = ProgressiveHedging(self.Instance, self.TestIdentifier, PHTreestructure)

        self.TestIdentifier.NrScenario = OldNrScenar
    def Run(self):

        self.GetHeuristicSetup()

        #solution = self.RunPH()

        self.ProgressiveHedging.InitTrace()
        # self.ProgressiveHedging.Run()
        self.ProgressiveHedging.CurrentSolution = [None for w in self.ProgressiveHedging.ScenarioNrSet]
        self.PrintOnlyFirstStagePreviousValue = Constants.PrintOnlyFirstStageDecision
        if Constants.PrintOnlyFirstStageDecision:
            Constants.PrintOnlyFirstStageDecision = False

        self.ProgressiveHedging.CurrentIteration = 0
        while self.ProgressiveHedging.CurrentIteration < 5 or not self.ProgressiveHedging.ComputeConvergenceY() <= 1.0:
            self.ProgressiveHedging.SolveScenariosIndependently()
            if self.ProgressiveHedging.CurrentIteration == -1:
                treestructure = [1, 200] + [1] * (self.Instance.NrTimeBucket - 1) + [0]
                self.TestIdentifier.Model = Constants.ModelYQFix
                chosengeneration = self.TestIdentifier.ScenarioSampling
                self.ScenarioGeneration = "RQMC"
                solution, mipsolver = self.Solver.MRP(treestructure, False, recordsolveinfo=True)
                self.ProgressiveHedging.GivenSetup = [[solution.Production[0][t][p] for p in self.Instance.ProductSet]
                                                      for t in self.Instance.TimeBucketSet]

                self.ProgressiveHedging.CurrentImplementableSolution= Solution.GetEmptySolution(self.Instance)

                self.ProgressiveHedging.CurrentImplementableSolution.ProductionQuantity = \
                                [[[solution.ProductionQuantity[0][t][p] for p in self.Instance.ProductSet]
                                                                      for t in self.Instance.TimeBucketSet]
                                                                     for w in self.ProgressiveHedging.ScenarioNrSet]

                self.ProgressiveHedging.CurrentImplementableSolution.Production = \
                    [[[solution.Production[0][t][p] for p in self.Instance.ProductSet]
                      for t in self.Instance.TimeBucketSet]
                     for w in self.ProgressiveHedging.ScenarioNrSet]


                self.ProgressiveHedging.CurrentImplementableSolution.Consumption = \
                    [ [[[solution.Consumption[0][t][p][q] for q in self.Instance.ProductSet]
                            for p in self.Instance.ProductSet]
                      for t in self.Instance.TimeBucketSet]
                     for w in self.ProgressiveHedging.ScenarioNrSet]
                solution= self.ProgressiveHedging.CurrentImplementableSolution

            else:
                solution = self.ProgressiveHedging.CreateImplementableSolution()
                self.ProgressiveHedging.CurrentImplementableSolution = solution


            self.ProgressiveHedging.PreviousImplementableSolution = copy.deepcopy(self.ProgressiveHedging.CurrentImplementableSolution)





            self.ProgressiveHedging.CurrentIteration += 1
            if self.ProgressiveHedging.CurrentIteration == 1:
                self.ProgressiveHedging.LagrangianMultiplier = 0.0001
            # if self.ProgressiveHedging.CurrentIteration >= 2:
            #     self.ProgressiveHedging.UpdateMultipler()
            self.ProgressiveHedging.UpdateLagragianMultipliers()

            #Just for the printing:
            self.ProgressiveHedging.CheckStopingCriterion()

            if Constants.Debug:
                self.ProgressiveHedging.PrintCurrentIteration()

        self.GivenSetup = [[solution.Production[0][t][p] for p in self.Instance.ProductSet] for t in
                           self.Instance.TimeBucketSet]  # solution.Production[0][t][p]

        Constants.PrintOnlyFirstStageDecision = self.PrintOnlyFirstStagePreviousValue

        self.RunSDDP()

      #  return solution

    def RunSDDP(self):
     #   print("RUN SDDP")
        treestructure = []
        self.SDDPSolver = SDDP(self.Instance, self.TestIdentifier, self.TreeStructure)

        #Mke sure SDDP do not unter in preliminary stage (at the end of the preliminary stage, SDDP would change the setup to bynary)
        Constants.SDDPGenerateCutWith2Stage = False
        Constants.SolveRelaxationFirst = False
        Constants.SDDPRunSigleTree = False


        self.SDDPSolver.HeuristicSetupValue = self.GivenSetup

        self.SDDPSolver.Run()
       # return self.SDDPSolver.CreateSolutionOfAllInSampleScenario()




    def GetHeuristicSetup(self):
        print("Get Heuristic Setups")
        treestructure = [1, 200] + [1] * (self.Instance.NrTimeBucket - 1) + [0]
        self.TestIdentifier.Model = Constants.ModelYQFix
        chosengeneration = self.TestIdentifier.ScenarioSampling
        self.ScenarioGeneration = "RQMC"
        solution, mipsolver = self.Solver.MRP(treestructure, False, recordsolveinfo=True)
        self.GivenSetup = [[solution.Production[0][t][p] for p in self.Instance.ProductSet] for t in
                           self.Instance.TimeBucketSet]  # solution.Production[0][t][p]
        self.ScenarioGeneration = chosengeneration
        self.TestIdentifier.Model = Constants.ModelYFix
        self.TestIdentifier.Method = Constants.Hybrid


 #   def InitTrace(self):
 #       if Constants.PrintSDDPTrace:
 #           self.TraceFile = open(self.TraceFileName, "w")
 #           self.TraceFile.write("Start the Hybrid Progressive Hedging / SDDP algorithm \n")
 #           self.TraceFile.close()

 #   def WriteInTraceFile(self, string):
 #       if Constants.PrintSDDPTrace:
 #           self.TraceFile = open(self.TraceFileName, "a")
 #           self.TraceFile.write(string)
 #           self.TraceFile.close()