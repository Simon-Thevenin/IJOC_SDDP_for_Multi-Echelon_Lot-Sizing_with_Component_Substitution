# This class contains the attributes and methods allowing to define the progressive hedging algorithm.
from ScenarioTree import ScenarioTree
from Constants import Constants
from MIPSolver import MIPSolver
from Solution import Solution

import time

class ProgressiveHedging(object):

    def __init__(self, instance, testidentifier, treestructure):
        self.Instance = instance
        self.TestIdentifier = testidentifier
        self.TreeStructure = treestructure

        self.GenerateScenarios()

        self.LagrangianMultiplier = 0
        self.CurrentIteration = 0
        self.StartTime = time.time()
        self.BuildMIPs()

    #This function creates the scenario tree
    def GenerateScenarios(self):
        print("to be defined")
        #Build the scenario tree
        print(self.TreeStructure)
        self.ScenarioTree = ScenarioTree(self.Instance, self.TreeStructure, self.TestIdentifier.ScenarioSeed,
                                         scenariogenerationmethod=self.TestIdentifier.ScenarioSampling,
                                         model=Constants.ModelYFix)

        self.ScenarioSet = self.ScenarioTree.GetAllScenarios(False)
        self.ScenarioNrSet = range(len(self.ScenarioSet))
        self.SplitScenrioTree()

    def BuildMIPs(self):
        #Build the mathematicals models (1 per scenarios)
        print("To be implemented")
        self.MIPSolvers = [MIPSolver(self.Instance, Constants.ModelYFix, self.SplitedScenarioTree[w],
                                     implicitnonanticipativity=True)
                           for w in self.ScenarioNrSet]

        for w in self.ScenarioNrSet:
            self.MIPSolvers[w].BuildModel()


    def SplitScenrioTree(self):

        treestructure = [1] + [1] * self.Instance.NrTimeBucket + [0]
        self.SplitedScenarioTree = [None for s in self.ScenarioNrSet]

        for scenarionr in self.ScenarioNrSet:
            scenario = self.ScenarioSet[scenarionr]
            self.SplitedScenarioTree[scenarionr] = ScenarioTree(self.Instance, treestructure, 0,
                                                              givenfirstperiod=scenario.Demands,
                                                              scenariogenerationmethod=self.TestIdentifier.ScenarioSampling,
                                                              model=Constants.ModelYFix)

            justotest = self.SplitedScenarioTree[scenarionr].GetAllScenarios(False)
            justotest[0].DisplayScenario()

    def CheckStopingCriterion(self):
        print("Check stopping criterion")
        duration = time.time() - self.StartTime
        timelimitreached = duration > Constants.AlgorithmTimeLimit
        iterationlimitreached = self.CurrentIteration > Constants.PHIterationLimit
        result = timelimitreached or iterationlimitreached
        return result

    def SolveScenariosIndependently(self):
        print("To be implemented")
        #For each scenario
        for w in self.ScenarioNrSet:

            #Update the coeffient in the objective function
            self.UpdateLagrangianCoeff()

            #Solve the model.
            solution = self.MIPSolvers[w].Solve(True)
            solution.Print()

    def UpdateLagrangianCoeff(self):
        print("To be implemented")

    def GetScenariosAssociatedWithNode(self):
        print("To be implemented")

    def CreateImplementableSolution(self):
        print("To be implemented")

        solquantity = [[[-1 for p in self.Instance.ProductSet]
                            for t in self.Instance.TimeBucketSet]
                          for w in self.ScenarioNrSet]

        solproduction = [[[-1 for p in self.Instance.ProductSet]
                        for t in self.Instance.TimeBucketSet]
                       for w in self.ScenarioNrSet]

        solinventory = [[[-1 for p in self.Instance.ProductSet]
                        for t in self.Instance.TimeBucketSet]
                       for w in self.ScenarioNrSet]

        solbackorder = [[[-1 for p in self.Instance.ProductSet]
                        for t in self.Instance.TimeBucketSet]
                       for w in self.ScenarioNrSet]

        solconsumption = [[[-1 for c in self.Instance.ConsumptionSet]
                        for t in self.Instance.TimeBucketSet]
                       for w in self.ScenarioNrSet]

        #For each node on the tree
        for n in self.ScenarioTree.Nodes:
            scenarios = self.GetScenariosAssociatedWithNode(n)
            time = n.Time

            # Average the quantities, and setups for this nodes.
            qty = [sum(self.CurrentSolution[w].ProductionQuantity[time][p] for w in scenarios) \
                        / len(scenarios)
                       for p in self.Instance.ProductSet]

            for w in scenarios:
                for p in self.Instance.ProductSet:
                    solquantity[w][time][p]= qty[p]


        solution = Solution(self.Instance, solquantity, solproduction, solinventory, solbackorder,
                            solconsumption,  scenarios, self.DemandScenarioTree)

        return solution

    def UpdateLagragianMultipliers(self):
        print("To be implemented")

    #This function run the algorithm
    def Run(self):
        solution = None
        self.CurrentSolution = [None for w in self.ScenarioNrSet]
        while not self.CheckStopingCriterion():
            # Solve each scenario independentely
            self.SolveScenariosIndependently()

            # Create an implementable solution on the scenario tree
            self.CurrentSolution[w] = self.CreateImplementableSolution(True)

            # Update the lagrangian multiplier
            self.UpdateLagragianMultipliers()

            self.CurrentIteration += 1

        return