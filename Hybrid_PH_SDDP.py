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
        if self.TestIdentifier.HybridPHSetting == "Multiplier100":
            Constants.PHMultiplier = 100.0
        if self.TestIdentifier.HybridPHSetting == "Multiplier10":
            Constants.PHMultiplier = 10.0
        if self.TestIdentifier.HybridPHSetting == "Multiplier1":
            Constants.PHMultiplier = 1.0
        if self.TestIdentifier.HybridPHSetting == "Multiplier01":
            Constants.PHMultiplier = 0.1
        if self.TestIdentifier.HybridPHSetting == "Multiplier001":
            Constants.PHMultiplier = 0.01
        if self.TestIdentifier.HybridPHSetting == "Multiplier0001":
            Constants.PHMultiplier = 0.001
        if self.TestIdentifier.HybridPHSetting == "Multiplier00001":
            Constants.PHMultiplier = 0.0001
        if self.TestIdentifier.HybridPHSetting == "Multiplier0":
             Constants.PHMultiplier = 0.0

 #       self.TraceFileName = "./Temp/SDDPPHtrace_%s.txt" % (self.TestIdentifier.GetAsString())
#
        self.Solver = solver

        self.NrScenarioOnceYIsFix = self.TestIdentifier.NrScenario

        if not Constants.MIPBasedOnSymetricTree:
            if self.Instance.NrTimeBucket > 5:
             self.TestIdentifier.NrScenario = "all2"
            else:
                self.TestIdentifier.NrScenario = "all5"

        if self.Instance.NrTimeBucket > 5:
            self.TestIdentifier.NrScenario = "all2"

        PHTreestructure = solver.GetTreeStructure()

        self.ProgressiveHedging = ProgressiveHedging(self.Instance, self.TestIdentifier, PHTreestructure)

        self.TestIdentifier.NrScenario = self.NrScenarioOnceYIsFix


    #This function is the main loop of the hybrid progressive hedging/SDDP heuristic
    def Run(self):

        #self.GetHeuristicSetup()
        self.ProgressiveHedging.InitTrace()
        self.ProgressiveHedging.CurrentSolution = [None for w in self.ProgressiveHedging.ScenarioNrSet]
        self.PrintOnlyFirstStagePreviousValue = Constants.PrintOnlyFirstStageDecision
        if Constants.PrintOnlyFirstStageDecision:
            Constants.PrintOnlyFirstStageDecision = False

        self.ProgressiveHedging.CurrentIteration = 0
        stop = False
        while not stop:
            self.ProgressiveHedging.SolveScenariosIndependently()
            if False and self.ProgressiveHedging.CurrentIteration == -1:
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
                self.ProgressiveHedging.LagrangianMultiplier = Constants.PHMultiplier #0.0001

            self.ProgressiveHedging.UpdateLagragianMultipliers()

            #Just for the printing:
            stop = self.ProgressiveHedging.CheckStopingCriterion()

            if Constants.Debug:
                self.ProgressiveHedging.PrintCurrentIteration()

        self.GivenSetup = [[solution.Production[0][t][p] for p in self.Instance.ProductSet] for t in
                           self.Instance.TimeBucketSet]  # solution.Production[0][t][p]

        Constants.PrintOnlyFirstStageDecision = self.PrintOnlyFirstStagePreviousValue

        self.RunSDDP()


    #This function runs SDDP for the current values of the setup
    def RunSDDP(self):
     #   print("RUN SDDP")
        self.SDDPSolver = SDDP(self.Instance, self.TestIdentifier, self.TreeStructure)

        #Mke sure SDDP do not unter in preliminary stage (at the end of the preliminary stage, SDDP would change the setup to bynary)
        Constants.SDDPGenerateCutWith2Stage = False
        Constants.SolveRelaxationFirst = False
        Constants.SDDPRunSigleTree = False

        self.SDDPSolver.HeuristicSetupValue = self.GivenSetup

        self.SDDPSolver.Run()
       # return self.SDDPSolver.CreateSolutionOfAllInSampleScenario()



    #This function generate the initial setup with the two-stage heuristic
    def GetHeuristicSetup(self):
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

