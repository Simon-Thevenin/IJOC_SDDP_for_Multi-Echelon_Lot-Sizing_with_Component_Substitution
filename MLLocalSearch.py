# This class contains the attributes and methods allowing to define the progressive hedging algorithm.
from ScenarioTree import ScenarioTree
from Constants import Constants
from MIPSolver import MIPSolver
from SDDP import SDDP
from sklearn import datasets, linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_absolute_error
import random

from ProgressiveHedging import ProgressiveHedging

from Solution import Solution

import copy
import time
import math

class MLLocalSearch(object):

    def __init__(self, instance, testidentifier, treestructure, solver):
        self.Instance = instance
        self.TestIdentifier = testidentifier
        self.TreeStructure = treestructure

        self.TraceFileName = "./Temp/MLLocalSearch%s.txt" % (self.TestIdentifier.GetAsString())
#
        self.Solver = solver
        self.TestedSetup = []
        self.CostToGoOfTestedSetup = []

        self.BestSolution = None
        self.BestSolutionCost = Constants.Infinity
        self.BestSolutionSafeUperBound = Constants.Infinity
        self.SDDPSolver = SDDP(self.Instance, self.TestIdentifier)
        self.SDDPSolver.HasFixedSetup = True
        self.SDDPSolver.IsIterationWithConvergenceTest = False
        # self.SDDPSolver.Run()
        self.SDDPSolver.GenerateSAAScenarios()

        # Mke sure SDDP do not unter in preliminary stage (at the end of the preliminary stage, SDDP would change the setup to bynary)
        Constants.SDDPGenerateCutWith2Stage = False
        Constants.SolveRelaxationFirst = False
        Constants.SDDPRunSigleTree = False

        treestructure = [1, 10] + [1] * (self.Instance.NrTimeBucket - 1) + [0]

        solution, self.SingleScenarioMipSolver = self.Solver.MRP(treestructure)

        self.CurrentScenarioSeed = 0

        self.Iteration = 0
        self.CurrentTolerance = 0.25
        self.InitTrace()

    def updateRecord(self, solution):
        if solution.TotalCost < self.BestSolutionCost:
            self.BestSolutionCost = solution.TotalCost
            self.BestSolutionSafeUperBound = max( self.SDDPSolver.CurrentSafeUpperBound, self.SDDPSolver.CurrentLowerBound)
            self.BestSolution = solution

    def trainML(self):
        self.Regr = linear_model.LinearRegression()
        self.Regr.fit(self.TestedSetup, self.CostToGoOfTestedSetup)
        self.poly = PolynomialFeatures(degree=2)
        X_poly = self.poly.fit_transform(self.TestedSetup)

        # poly.fit(X_poly, self.CostToGoOfTestedSetup)
        self.lin2 = LinearRegression()
        self.lin2.fit(X_poly, self.CostToGoOfTestedSetup)

        self.clf = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15))
        self.clf.fit(self.TestedSetup, self.CostToGoOfTestedSetup)

        self.reg = linear_model.Ridge(alpha=.5)
        self.reg.fit(self.TestedSetup, self.CostToGoOfTestedSetup)

    def GenerateOutSample(self):
        self.outofsampletest = []
        self.outofsamplecost = []
        for i in range(1, 2):
            self.GivenSetup1D, self.GivenSetup2D = self.GetRandomSetups()
            solution = self.RunSDDP()

            self.outofsampletest.append(self.GivenSetup1D)
            self.outofsamplecost.append(solution.TotalCost - solution.SetupCost)

    def RunSDDPAndAddToTraining(self):
        solution = self.RunSDDP()

        self.TestedSetup.append(self.GivenSetup1D)
        self.CostToGoOfTestedSetup.append(solution.TotalCost - solution.SetupCost)

        self.updateRecord(solution)
    def Run(self):

        #self.GetHeuristicSetup()

        #Sample a set of 10 scenario.

        start = time.time()
        duration = 0
        #Initialization
        for i in range(1,2):
            self.GivenSetup1D, self.GivenSetup2D = self.GetRandomSetups()
            self.RunSDDPAndAddToTraining()


        self.trainML()
        self.GenerateOutSample()

        curentsolution = copy.deepcopy(self.BestSolution)

        while( duration < Constants.AlgorithmTimeLimit):
            if(self.Iteration >= 1):
                self.WriteInTraceFile("use heuristic setups")
                self.GivenSetup2D = self.Descent(curentsolution)
            else:
                self.GivenSetup2D = self.GetHeuristicSetup()

            self.GivenSetup1D = [self.GivenSetup2D[t][p] for p in self.Instance.ProductSet for t in
                                 self.Instance.TimeBucketSet]
            self.RunSDDPAndAddToTraining()

            self.WriteInTraceFile("considered setup : %r , ML cost :%r actual cost :%r \n" % (self.GivenSetup1D, self.PredictForSetups(self.GivenSetup2D) , self.CostToGoOfTestedSetup[-1]))

            self.trainML()

            if( self.Iteration == 10 ):
                self.CurrentTolerance = Constants.AlgorithmOptimalityTolerence
                self.BestSolutionCost = Constants.Infinity
                self.BestSolutionSafeUperBound = Constants.Infinity
                self.BestSolution = None

            # print("-------------------------Linear Regression-----------------------------")
            # insampleprediction = self.Regr.predict(self.TestedSetup)
            # outsampleprediction = self.Regr.predict(self.outofsampletest)
            # print("self.TestedSetup : %r "%self.TestedSetup)
            # print("self.CostToGoOfTestedSetup : %r "%self.CostToGoOfTestedSetup)
            # print("insampleprediction : %r " % insampleprediction)
            # print("insample absolute error %r"% mean_absolute_error(self.CostToGoOfTestedSetup, insampleprediction) )
            # print("outsampleprediction : %r " % outsampleprediction)
            # print("outsample absolute error %r" % mean_absolute_error(self.outofsamplecost, outsampleprediction))
            #
            # print("-------------------------Polynomial Regression-----------------------------")
            # X_poly = self.poly.fit_transform(self.TestedSetup)
            # insamplepredictionpoly = self.lin2.predict(X_poly)
            # X_poly = self.poly.fit_transform(self.outofsampletest)
            # outsamplepredictionpoly = self.lin2.predict(X_poly)
            # print("insamplepredictionpoly : %r " % insamplepredictionpoly)
            # print("insample absolute error predictionpoly %r" % mean_absolute_error(self.CostToGoOfTestedSetup, insamplepredictionpoly))
            # print("outsamplepredictionpoly : %r " % insamplepredictionpoly)
            # print("outsample absolute error predictionpoly %r" % mean_absolute_error(self.outofsamplecost,
            #                                                                          outsamplepredictionpoly))
            #
            # print("------------------------- Ridge -----------------------------")
            # insamplepredictionRidge = self.reg.predict(self.TestedSetup)
            # outsamplepredictionRidge = self.reg.predict(self.outofsampletest)
            # print("self.TestedSetup : %r " % self.TestedSetup)
            # print("self.CostToGoOfTestedSetup : %r " % self.CostToGoOfTestedSetup)
            # print("insampleprediction : %r " % insamplepredictionRidge)
            # print("insample absolute error %r" % mean_absolute_error(self.CostToGoOfTestedSetup,
            #                                                          insamplepredictionRidge))
            # print("outsampleprediction : %r " % outsamplepredictionRidge)
            # print("outsample absolute error %r" % mean_absolute_error(self.outofsamplecost, outsamplepredictionRidge))


            insamplepredictionNN = self.clf.predict(self.TestedSetup)
            outsamplepredictionNN = self.clf.predict(self.outofsampletest)
            if Constants.Debug:
                print("-------------------------Neural net-----------------------------")
                print("self.TestedSetup : %r " % self.TestedSetup)
                print("self.CostToGoOfTestedSetup : %r " % self.CostToGoOfTestedSetup)
                print("insampleprediction : %r " % insamplepredictionNN)
                print("insample absolute error %r" % mean_absolute_error(self.CostToGoOfTestedSetup, insamplepredictionNN))
                print("outsampleprediction : %r " % outsamplepredictionNN)
                print("outsample absolute error %r" % mean_absolute_error(self.outofsamplecost, outsamplepredictionNN))

            self.WriteInTraceFile( "outsample absolute error %r" % mean_absolute_error(self.outofsamplecost, outsamplepredictionNN) )



            self.Iteration += 1

            end = time.time()
            duration = end - start

        self.GivenSetup2D = self.BestSolution.Production[0]
        self.SDDPSolver.IsIterationWithConvergenceTest = True
        self.BestSolution = self.RunSDDP()

        return self.BestSolution

    def GetCostBasedML(self, setups):
        return self.clf.predict(setups)


    def Descent(self, initialsolution):
        currentsolution = initialsolution

        self.UseTabu = True
        self.DescentBestCost = Constants.Infinity
        self.DescentBestMove = ("N", -1, -1)

        self.TabuCurrentSolLB = Constants.Infinity
        self.TabuBestSolLB = Constants.Infinity
        self.TabuBestSolLB = Constants.Infinity
        self.TabuBestSolPredictedUB = Constants.Infinity
        self.TabuBestSol = None
       # currentsolution.Production[0] = [[random.randint(0, 1) for p in self.Instance.ProductSet] for t in
       #                                      self.Instance.TimeBucketSet]

        iterationtabu = [[0 for p in self.Instance.ProductSet] for t in self.Instance.TimeBucketSet]
        curentiterationLS = 0
        while ( not self.UseTabu and self.DescentBestMove[0] <> "") or (self.UseTabu and (self.TabuBestSol is None or curentiterationLS < 100)):
                self.TabuCurrentPredictedUB = Constants.Infinity
                self.TabuCurrentSolLB = Constants.Infinity
                self.DescentBestMove = ("", -1, -1)
                for p in self.Instance.ProductSet:
                     for t in self.Instance.TimeBucketSet:
                        if iterationtabu[t][p] <= curentiterationLS:

                            if currentsolution.Production[0][t][p] == 1:
                                #Move earlier
                                if t> 0:
                                    self.EvaluateMoveEarlier(currentsolution.Production[0], t, p)
                                if t<self.Instance.NrTimeBucket-1:
                                    self.EvaluateMoveLater(currentsolution.Production[0], t, p)
                                self.EvaluateRemove(currentsolution.Production[0], t, p)
                            else:
                                self.EvaluateAdd(currentsolution.Production[0], t, p)



                #perform the best move
                if self.UseTabu:
                    self.DescentBestMove = self.TabuBestMove

                t = self.DescentBestMove[1]
                p = self.DescentBestMove[2]

                tabulistsize = 5
              #  print("Move: %r %r %r"%(self.DescentBestMove[0], t, p))
                if self.DescentBestMove[0] == "E":
                    currentsolution.Production[0][t][p] = 0
                    currentsolution.Production[0][t - 1][p] = 1
                    iterationtabu[t-1][p] = curentiterationLS + tabulistsize
                    iterationtabu[t][p] = curentiterationLS + tabulistsize
                if self.DescentBestMove[0] == "L":
                    currentsolution.Production[0][t][p] = 0
                    currentsolution.Production[0][t + 1][p] = 1
                    iterationtabu[t + 1][p] = curentiterationLS + tabulistsize
                    iterationtabu[t][p] = curentiterationLS + tabulistsize
                if self.DescentBestMove[0] == "A":
                    currentsolution.Production[0][t][p] = 1
                    iterationtabu[t][p] = curentiterationLS + tabulistsize
                if self.DescentBestMove[0] == "R":
                    currentsolution.Production[0][t][p] = 0
                    iterationtabu[t][p] = curentiterationLS + tabulistsize

                cost = self.PredictForSetups(currentsolution.Production[0])
                lb = self.GetCurrentLowerBound(currentsolution.Production[0])
                newrecord = False
              #  print(cost)
              #  print(self.GetCurrentLowerBound(currentsolution.Production[0]))

                if self.TabuBestSolLB < self.BestSolutionSafeUperBound:
                    if lb < self.BestSolutionSafeUperBound and cost < self.TabuBestCost:
                        newrecord = True
                       # print("a")
                else:
                    if( lb <   self.TabuBestSolLB ):
                        newrecord = True
                      #  print("b")

                if newrecord:
                     #   print("New record!!!")
                        self.TabuBestCost = cost
                        self.TabuBestSolLB = lb
                        self.TabuBestSol = copy.deepcopy(currentsolution)

                curentiterationLS += 1


        return self.TabuBestSol.Production[0]




    def UpdateDescentRecord(self, move, t, p, cost, setups):
      #  if cost < self.DescentBestCost and self.GetCurrentLowerBound(setups) < self.BestSolutionSafeUperBound:
      #      self.DescentBestCost = cost
      #      self.DescentBestMove = ( move, t, p )
           # print("Evaluate %r %r %r "%(move, t, p) )

       # if  self.UseTabu :
            lb = self.GetCurrentLowerBound(setups)
            newrecord = False

            if self.TabuCurrentSolLB < self.BestSolutionSafeUperBound:
                if lb < self.BestSolutionSafeUperBound and cost < self.TabuCurrentPredictedUB:
                    newrecord = True
            else:
                if (lb < self.TabuCurrentSolLB):
                    newrecord = True

            if newrecord:
                self.TabuCurrentPredictedUB = cost
                self.TabuCurrentSolLB = lb
                self.TabuBestMove = ( move, t, p )

    def EvaluateMoveEarlier(self, currentsetups, t, p ):
        previousvaluet = currentsetups[t][p]
        previousvalueprev = currentsetups[t - 1][p]
        currentsetups[t][p] = 0
        currentsetups[t - 1][p] = 1
        cost = self.PredictForSetups(currentsetups)
        self.UpdateDescentRecord("E", t, p, cost, currentsetups)
        currentsetups[t][p] = previousvaluet
        currentsetups[t - 1][p] = previousvalueprev
        return cost

    def EvaluateMoveLater(self, currentsetups, t, p):
        previousvaluet = currentsetups[t][p]
        previousvalueprev = currentsetups[t + 1][p]
        currentsetups[t][p] = 0
        currentsetups[t + 1][p] = 1
        cost = self.PredictForSetups(currentsetups)
        self.UpdateDescentRecord("L", t, p, cost, currentsetups)
        currentsetups[t][p] = previousvaluet
        currentsetups[t + 1][p] = previousvalueprev
        return cost

    def EvaluateAdd(self, currentsetups, t, p):
        previousvaluet = currentsetups[t][p]
        currentsetups[t][p] = 1
        cost = self.PredictForSetups(currentsetups)
        self.UpdateDescentRecord("A", t, p, cost, currentsetups)
        currentsetups[t][p] = previousvaluet
        return cost

    def EvaluateRemove(self, currentsetups, t, p):
        previousvaluet = currentsetups[t][p]
        currentsetups[t][p] = 0
        cost = self.PredictForSetups(currentsetups)
        self.UpdateDescentRecord("R", t, p, cost, currentsetups)
        currentsetups[t][p] = previousvaluet
        return cost

    def PredictForSetups(self, setups):
        setupcost = sum(setups[t][p] * self.Instance.SetupCosts[p] for t in self.Instance.TimeBucketSet  for p in self.Instance.ProductSet)

        setups1D = [[setups[t][p] for p in self.Instance.ProductSet for t in
                             self.Instance.TimeBucketSet]]
        approxcosttogo = self.GetCostBasedML(setups1D)
        return setupcost + approxcosttogo




    def RunSDDP(self):
       ## print("RUN SDDP")

        self.SDDPSolver.WriteInTraceFile("__________________New run of SDDP ____________________ \n")

        self.SDDPSolver.HeuristicSetupValue = self.GivenSetup2D



        if self.Iteration >= 1:
            self.SDDPSolver.ForwardStage[0].ChangeSetupToValueOfTwoStage()

        stop = False
        lastbeforestop = False

        self.SDDPSolver.CurrentForwardSampleSize = self.TestIdentifier.NrScenarioForward
        while (not stop ):
            self.SDDPSolver.GenerateTrialScenarios()
            self.SDDPSolver.ForwardPass()
            self.SDDPSolver.ComputeCost()
            self.SDDPSolver.UpdateLowerBound()
            self.SDDPSolver.UpdateUpperBound()
            self.SDDPSolver.BackwardPass()
            self.SDDPSolver.CurrentIteration = self.SDDPSolver.CurrentIteration + 1
            stop = self.CheckStopingSDDP()




        self.SDDPSolver.ForwardStage[0].ProductionValue = [[[self.SDDPSolver.HeuristicSetupValue[t][p]
                                                             for p in self.Instance.ProductSet]
                                                            for t in self.Instance.TimeBucketSet]
                                                           for w in range(len(self.SDDPSolver.CompleteSetOfSAAScenario))]




        solution =  self.SDDPSolver.CreateSolutionAtFirstStage()
        solution.TotalCost = self.SDDPSolver.CurrentExpvalueUpperBound

        if (self.SDDPSolver.IsIterationWithConvergenceTest):
            self.SDDPSolver.LastExpectedCostComputedOnAllScenario = self.SDDPSolver.CurrentExpvalueUpperBound
            solution.SDDPExpUB = self.SDDPSolver.CurrentExpvalueUpperBound

        return solution

    def GetHeuristicSetup(self):
      #  print("Get Heuristic Setups")
        treestructure = [1, 200] + [1] * (self.Instance.NrTimeBucket - 1) + [0]
        self.TestIdentifier.Model = Constants.ModelYQFix
        chosengeneration = self.TestIdentifier.ScenarioSampling
        self.ScenarioGeneration = "RQMC"
        solution, mipsolver = self.Solver.MRP(treestructure, False, recordsolveinfo=True)
        GivenSetup = [[solution.Production[0][t][p] for p in self.Instance.ProductSet] for t in
                           self.Instance.TimeBucketSet]  # solution.Production[0][t][p]
        self.ScenarioGeneration = chosengeneration
        self.TestIdentifier.Model = Constants.ModelYFix
        self.TestIdentifier.Method = Constants.MLLocalSearch
        return GivenSetup


    def InitTrace(self):
        if Constants.PrintSDDPTrace:
            self.TraceFile = open(self.TraceFileName, "w")
            self.TraceFile.write("Start the MLLocal search \n")
            self.TraceFile.close()

    def WriteInTraceFile(self, string):
        if Constants.PrintSDDPTrace:
            self.TraceFile = open(self.TraceFileName, "a")
            self.TraceFile.write(string)
            self.TraceFile.close()

    def GetSetupForSingleScenario(self):
        treestructure = [1, 10] + [1] * (self.Instance.NrTimeBucket - 1) + [0]
        self.CurrentScenarioSeed += 1
        self.TestIdentifier.Model = Constants.ModelYFix
        scenariotree = ScenarioTree(self.Instance, treestructure, self.CurrentScenarioSeed,
                                            model=Constants.ModelYFix )

        self.SingleScenarioMipSolver.ModifyMipForScenarioTree(scenariotree)

        solution = self.SingleScenarioMipSolver.Solve()

        print("scenario:%r"%self.SingleScenarioMipSolver.Scenarios[0].Demands)
        self.GivenSetup1D = [solution.Production[0][t][p] for p in self.Instance.ProductSet for t in
                           self.Instance.TimeBucketSet]
        self.GivenSetup2D = [[solution.Production[0][t][p] for p in self.Instance.ProductSet] for t in
                             self.Instance.TimeBucketSet]
        # solution.Production[0][t][p]
        return self.GivenSetup1D, self.GivenSetup2D


    def GetRandomSetups(self):

        self.GivenSetup2D = [[random.randint(0,1) for p in self.Instance.ProductSet] for t in
                             self.Instance.TimeBucketSet]

        self.GivenSetup1D = [self.GivenSetup2D[t][p] for p in self.Instance.ProductSet for t in
                             self.Instance.TimeBucketSet]

        return self.GivenSetup1D, self.GivenSetup2D

    def CheckStopingSDDP(self):
        convergencecriterion = Constants.Infinity
        c = Constants.Infinity
        if self.SDDPSolver.CurrentLowerBound > 0:
            convergencecriterion = float(self.SDDPSolver.CurrentUpperBound) / float(self.SDDPSolver.CurrentLowerBound) \
                                   - (1.96 * math.sqrt(float(self.SDDPSolver.VarianceForwardPass) \
                                                       / float(self.SDDPSolver.CurrentNrScenario)) \
                                      / float(self.SDDPSolver.CurrentLowerBound))

            c = (1.96 * math.sqrt(float(self.SDDPSolver.VarianceForwardPass) / float(self.SDDPSolver.CurrentNrScenario)) \
                 / float(self.SDDPSolver.CurrentLowerBound))

        delta = Constants.Infinity
        if self.SDDPSolver.CurrentLowerBound > 0:
            delta = 3.92 * math.sqrt(float(self.SDDPSolver.VarianceForwardPass) / float(self.SDDPSolver.CurrentNrScenario)) \
                    / float(self.SDDPSolver.CurrentLowerBound)

        self.SDDPSolver.WriteInTraceFile("Iteration SDDP for ML Descent LB: % r, (exp UB: % r - safe ub: %r), variance: %r, convergencecriterion: %r, delta: %r \n" % (
        self.SDDPSolver.CurrentLowerBound, self.SDDPSolver.CurrentExpvalueUpperBound, self.SDDPSolver.CurrentSafeUpperBound, self.SDDPSolver.VarianceForwardPass, convergencecriterion, delta))


        if(convergencecriterion <= 1 and delta > self.CurrentTolerance and  self.SDDPSolver.NrIterationWithoutLBImprovment>0):
            self.SDDPSolver.CurrentForwardSampleSize = max(self.SDDPSolver.CurrentForwardSampleSize * 2, self.SDDPSolver.NrScenarioSAA)




        return self.SDDPSolver.CurrentLowerBound > self.BestSolutionSafeUperBound \
                or (convergencecriterion <= 1 and delta <= self.CurrentTolerance)#

    def GetCurrentLowerBound(self, setups2D):
        self.SDDPSolver.HeuristicSetupValue = setups2D
        self.SDDPSolver.ForwardStage[0].ChangeSetupToValueOfTwoStage()
        self.SDDPSolver.ForwardStage[0].RunForwardPassMIP()
      #  print("tested setup: %r"%setups2D )

        self.SDDPSolver.ForwardStage[0].ComputePassCost()
      #  print("cost of solution %r"%self.SDDPSolver.ForwardStage[0].PassCostWithAproxCosttoGo)
        return self.SDDPSolver.ForwardStage[0].PassCostWithAproxCosttoGo