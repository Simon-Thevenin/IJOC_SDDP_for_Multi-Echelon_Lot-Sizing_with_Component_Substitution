# This class contains the attributes and methods allowing to define the progressive hedging algorithm.
from ScenarioTree import ScenarioTree
from Constants import Constants
from MIPSolver import MIPSolver
from Solution import Solution

import time

class ProgressiveHedging(object):

    def __init__(self, instance, testidentifier, treestructure):

        if Constants.PrintOnlyFirstStageDecision:
            raise NameError("Progressive Hedging requires to print the full solution, set Constants.PrintOnlyFirstStageDecision to True")

        self.Instance = instance
        self.TestIdentifier = testidentifier
        self.TreeStructure = treestructure

        self.GenerateScenarios()

        self.LagrangianMultiplier = 0.1
        self.LagrangianQuantity = [[[0 for p in self.Instance.ProductSet]
                                       for t in self.Instance.TimeBucketSet]
                                       for w in self.ScenarioNrSet]

        self.LagrangianProduction = [[[0 for p in self.Instance.ProductSet]
                                       for t in self.Instance.TimeBucketSet]
                                       for w in self.ScenarioNrSet]

        self.LagrangianConsumption = [[[[0 for p in self.Instance.ProductSet]
                                           for q in self.Instance.ProductSet]
                                           for t in self.Instance.TimeBucketSet]
                                           for w in self.ScenarioNrSet]

        self.CurrentIteration = 0
        self.StartTime = time.time()
        self.BuildMIPs()

    #This function creates the scenario tree
    def GenerateScenarios(self):
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
        #For each scenario
        for w in self.ScenarioNrSet:

            #Update the coeffient in the objective function
            self.UpdateLagrangianCoeff(w)

            #Solve the model.

            self.CurrentSolution[w] = self.MIPSolvers[w].Solve(True)
            #self.CurrentSolution[w].Print()


    def UpdateLagrangianCoeff(self, scenario):
        variables = []
        mipsolver = self.MIPSolvers[scenario]
        for t in self.Instance.TimeBucketSet:
                for p in self.Instance.ProductSet:
                    variable = mipsolver.GetIndexQuantityVariable(p, t, 0)
                    coeff = mipsolver.GetQuantityCoeff(p, t, 0) + self.LagrangianQuantity[scenario][t][p]
                    variables.append((variable, coeff))

                    variable = mipsolver.GetIndexProductionVariable(p, t, 0)
                    coeff = mipsolver.GetProductionCefficient(p, t, 0) + self.LagrangianProduction[scenario][t][p]
                    variables.append((variable, coeff))

                for c in self.Instance.ConsumptionSet:
                        variable = mipsolver.GetIndexConsumptionVariable(c[1], c[0], t, 0)
                        coeff = mipsolver.GetConsumptionCoeff(c[1], c[0], t, 0) + self.LagrangianConsumption[scenario][t][c[1]][c[0]]
                        variables.append((variable, coeff))

        print("New coeff: %r"%variables)
        mipsolver.Cplex.objective.set_linear(variables)

    def GetScenariosAssociatedWithNode(self, node):
        scenarios = node.Scenarios
        result = []
        for s in scenarios:
            result.append(self.ScenarioSet.index(s))

        return result

    def CreateImplementableSolution(self):
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

        solconsumption = [[[[-1 for p in self.Instance.ProductSet]
                           for q in self.Instance.ProductSet]
                        for t in self.Instance.TimeBucketSet]
                       for w in self.ScenarioNrSet]

        #For each node on the tree
        for n in self.ScenarioTree.Nodes:
            if n.Time >= 0:
                scenarios = self.GetScenariosAssociatedWithNode(n)
                time = n.Time

                # Average the quantities, and setups for this nodes.
                if time < self.Instance.NrTimeBucket:
                    qty = [sum(self.CurrentSolution[w].ProductionQuantity[0][time][p] for w in scenarios) \
                           / len(scenarios) for p in self.Instance.ProductSet]

                    prod = [sum(self.CurrentSolution[w].Production[0][time][p] for w in scenarios)
                            / len(scenarios) for p in self.Instance.ProductSet]

                    cons = [[sum(self.CurrentSolution[w].Consumption[0][time][p][q] for w in scenarios)
                            / len(scenarios) for p in self.Instance.ProductSet]
                            for q in self.Instance.ProductSet]

                    for w in scenarios:
                        for p in self.Instance.ProductSet:
                            solproduction[w][time][p] = int(round(prod[p]))
                            solquantity[w][time][p] = qty[p]
                            for q in self.Instance.ProductSet:
                                solconsumption[w][time][q][p] = cons[p][q]



        solution = Solution(self.Instance, solquantity, solproduction, solinventory, solbackorder,
                            solconsumption,  self.ScenarioSet, self.ScenarioTree)


        return solution

    def UpdateLagragianMultipliers(self):
        for w in self.ScenarioNrSet:
            for t in self.Instance.TimeBucketSet:
                for p in self.Instance.ProductSet:
                    self.LagrangianQuantity[w][t][p] += self.LagrangianMultiplier \
                                                        * (self.CurrentSolution[w].ProductionQuantity[0][t][p] \
                                                            - self.CurrentImplementableSolution.ProductionQuantity[w][t][p])

                    self.LagrangianProduction[w][t][p] += self.LagrangianMultiplier \
                                                          * (self.CurrentSolution[w].Production[0][t][p] \
                                                            - self.CurrentImplementableSolution.Production[w][t][p])


                    for q in self.Instance.ProductSet:
                        self.LagrangianConsumption[w][t][p][q] +=  self.LagrangianMultiplier \
                                                        * (self.CurrentSolution[w].Consumption[0][t][p][q]\
                                                            - self.CurrentImplementableSolution.Consumption[w][t][p][q])


    def PrintCurrentIteration(self):
        print("----------------Independent solutions--------------")
        for w in self.ScenarioNrSet:
            self.CurrentSolution[w].Print()
            print("+++++++++++++++++++++")
        print("-----------IMPLEMENTABLE: -------------------------")
        self.CurrentImplementableSolution.Print()
        print("---------------------------------------------------")


    #This function run the algorithm
    def Run(self):
        solution = None
        self.CurrentSolution = [None for w in self.ScenarioNrSet]
        while not self.CheckStopingCriterion():
            # Solve each scenario independentely
            self.SolveScenariosIndependently()

            # Create an implementable solution on the scenario tree
            self.CurrentImplementableSolution = self.CreateImplementableSolution()

            # Update the lagrangian multiplier
            self.UpdateLagragianMultipliers()

            self.CurrentIteration += 1
        self.PrintCurrentIteration()

        return