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

        self.ProgressiveHedging = ProgressiveHedging(self.Instance, self.TestIdentifier, self.TreeStructure)

    def Run(self):

        #self.GetHeuristicSetup()

        #solution = self.RunPH()

        self.ProgressiveHedging.InitTrace()
        # self.ProgressiveHedging.Run()
        self.ProgressiveHedging.CurrentSolution = [None for w in self.ProgressiveHedging.ScenarioNrSet]
        Constants.PrintOnlyFirstStageDecision = False

        while not self.ProgressiveHedging.CheckStopingCriterion():
            self.ProgressiveHedging.SolveScenariosIndependently()
            solution = self.ProgressiveHedging.CreateImplementableSolution()

            self.GivenSetup = [[solution.Production[0][t][p] for p in self.Instance.ProductSet] for t in
                               self.Instance.TimeBucketSet]  # solution.Production[0][t][p]

            solution = self.RunSDDP()

            self.ProgressiveHedging.PreviousImplementableSolution = copy.deepcopy(self.ProgressiveHedging.CurrentImplementableSolution)
            self.ProgressiveHedging.CurrentImplementableSolution = solution

            self.ProgressiveHedging.PrintCurrentIteration()


            self.ProgressiveHedging.CurrentIteration += 1
            if self.ProgressiveHedging.CurrentIteration == 1:
                self.ProgressiveHedging.LagrangianMultiplier = 0.0001

            self.ProgressiveHedging.UpdateLagragianMultipliers()

        return solution

    def RunSDDP(self):
        print("RUN SDDP")
        self.SDDPSolver = SDDP(self.Instance, self.TestIdentifier)

        #Mke sure SDDP do not unter in preliminary stage (at the end of the preliminary stage, SDDP would change the setup to bynary)
        Constants.SDDPGenerateCutWith2Stage = False
        Constants.SolveRelaxationFirst = False
        Constants.SDDPRunSigleTree = False

        self.SDDPSolver.HeuristicSetupValue = self.GivenSetup
        self.SDDPSolver.Run()
        return self.SDDPSolver.CreateSolutionOfAllInSampleScenario()




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