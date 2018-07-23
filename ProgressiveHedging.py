# This class contains the attributes and methods allowing to define the progressive hedging algorithm.
from ScenarioTree import ScenarioTree
from Constants import Constants
from MIPSolver import MIPSolver
from Solution import Solution

import copy
import time
import math

class ProgressiveHedging(object):

    def __init__(self, instance, testidentifier, treestructure):

        if Constants.PrintOnlyFirstStageDecision:
            raise NameError("Progressive Hedging requires to print the full solution, set Constants.PrintOnlyFirstStageDecision to False")

        self.Instance = instance
        self.TestIdentifier = testidentifier
        self.TreeStructure = treestructure

        self.GenerateScenarios()

        self.LagrangianMultiplier = 0.1
        self.CurrentImplementableSolution = None
        self.TraceFileName = "./Temp/PHtrace_%s.txt" % (self.TestIdentifier.GetAsString())
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

        self.LinearLagQuantity = [[[0 for p in self.Instance.ProductSet]
                                    for t in self.Instance.TimeBucketSet]
                                   for w in self.ScenarioNrSet]

        self.LinearLagProduction = [[[0 for p in self.Instance.ProductSet]
                                      for t in self.Instance.TimeBucketSet]
                                     for w in self.ScenarioNrSet]

        self.LinearLagConsumption = [[[[0 for p in self.Instance.ProductSet]
                                        for q in self.Instance.ProductSet]
                                       for t in self.Instance.TimeBucketSet]
                                      for w in self.ScenarioNrSet]


        self.CurrentIteration = 0
        self.StartTime = time.time()
        self.BuildMIPs()

    #This function creates the scenario tree
    def GenerateScenarios(self):
        #Build the scenario tree
        if Constants.Debug:
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
        gap = Constants.Infinity

        if self.CurrentIteration > 0:
            gap = self.ComputeConvergence()

        convergencereached = gap < Constants.PHConvergenceTolerence

        duration = time.time() - self.StartTime
        timelimitreached = duration > Constants.AlgorithmTimeLimit
        iterationlimitreached = self.CurrentIteration > Constants.PHIterationLimit
        result = convergencereached or timelimitreached or iterationlimitreached

        if Constants.PrintSDDPTrace and self.CurrentIteration > 0:
            self.CurrentImplementableSolution.ComputeInventory()
            self.CurrentImplementableSolution.ComputeCost()

            print("Demands")
            for w in self.CurrentImplementableSolution.Scenarioset:
                print(w.Demands)

            dualconv = -1
            primconv = -1
            lpenalty = self.GetLinearPenalty()
            qpenalty = self.GetQuadraticPenalty()
            ratequad_lin = self.RateQuadLinear()
            ratechangeimplem = -1
            ratedualprimal = -1
            rateprimaldual = -1
            if self.CurrentIteration>1:
                primconv = self.GetPrimalConvergenceIndice()
                dualconv = self.GetDualConvergenceIndice()
                ratechangeimplem = self.RateLargeChangeInImplementable()
                rateprimaldual = self.RatePrimalDual()
                ratedualprimal = self.RateDualPrimal()



                print("completeCost %r"%(self.CurrentImplementableSolution.TotalCost + lpenalty + qpenalty))



            self.WriteInTraceFile("Iteration: %r Duration: %.2f Gap: %.2f UB:%.2f linear penalty:%.2f quadratic penalty: %.2f"
                                  " Multiplier:%.2f primal conv:%.2f dual conv:%.2f Rate Large Change(l): %.2f"
                                  " rate quad_lin(s):%.2f rateprimaldual(l<-):%.2f ratedualprimal(l->): %.2f \n"
                                  % (self.CurrentIteration, duration, gap, self.CurrentImplementableSolution.TotalCost,
                                     lpenalty, qpenalty, self.LagrangianMultiplier, primconv, dualconv, ratechangeimplem,
                                     ratequad_lin, rateprimaldual, ratedualprimal))

        return result


    def SolveScenariosIndependently(self):
        #For each scenario
        for w in self.ScenarioNrSet:

            #Update the coeffient in the objective function
            self.UpdateLagrangianCoeff(w)

            #Solve the model.

            self.CurrentSolution[w] = self.MIPSolvers[w].Solve(True)

            if self.CurrentIteration > 1:

                qp = sum(math.pow((self.CurrentSolution[w].ProductionQuantity[0][t][p]
                                         - self.CurrentImplementableSolution.ProductionQuantity[w][t][p]), 2)
                             for p in self.Instance.ProductSet
                             for t in self.Instance.TimeBucketSet)

                qp += sum(math.pow((self.CurrentSolution[w].Production[0][t][p]
                                          - self.CurrentImplementableSolution.Production[w][t][p]), 2)
                              for p in self.Instance.ProductSet
                              for t in self.Instance.TimeBucketSet)

                qp += sum(math.pow((self.CurrentSolution[w].Consumption[0][t][c[0]][c[1]]
                                          - self.CurrentImplementableSolution.Consumption[w][t][c[0]][c[1]]), 2)
                              for c in self.Instance.ConsumptionSet
                              for t in self.Instance.TimeBucketSet)

                qp *= self.LagrangianMultiplier * 0.5

                lp = sum( self.LinearLagQuantity[w][t][p] \
                             * (self.CurrentSolution[w].ProductionQuantity[0][t][p]
                                - self.CurrentImplementableSolution.ProductionQuantity[w][t][p])
                             for p in self.Instance.ProductSet
                             for t in self.Instance.TimeBucketSet)

                lp += sum(self.LinearLagProduction[w][t][p]
                              * (self.CurrentSolution[w].Production[0][t][p]
                                 - self.CurrentImplementableSolution.Production[w][t][p])
                              for p in self.Instance.ProductSet
                              for t in self.Instance.TimeBucketSet)

                lp += sum(self.LinearLagConsumption[0][t][c[0]][c[1]] \
                          * (self.CurrentSolution[w].Consumption[0][t][c[0]][c[1]]
                             - self.CurrentImplementableSolution.Consumption[w][t][c[0]][c[1]])
                              for c in self.Instance.ConsumptionSet
                              for t in self.Instance.TimeBucketSet)

                penalty = lp + qp

                lpconst = sum(- self.LinearLagQuantity[w][t][p] \
                              * (self.CurrentImplementableSolution.ProductionQuantity[w][t][p])
                              for p in self.Instance.ProductSet
                              for t in self.Instance.TimeBucketSet)

                lpconst += sum(-self.LinearLagProduction[w][t][p]
                               * (self.CurrentImplementableSolution.Production[w][t][p])
                               for p in self.Instance.ProductSet
                               for t in self.Instance.TimeBucketSet)

                lpconst += sum(- self.LinearLagConsumption[0][t][c[0]][c[1]] \
                               * (self.CurrentImplementableSolution.Consumption[w][t][c[0]][c[1]])
                               for c in self.Instance.ConsumptionSet
                               for t in self.Instance.TimeBucketSet)

                qpconst = sum(0.5 * self.LagrangianMultiplier \
                              * math.pow((self.CurrentImplementableSolution.ProductionQuantity[w][t][p]), 2)
                              for p in self.Instance.ProductSet
                              for t in self.Instance.TimeBucketSet)

                qpconst += sum(0.5 * self.LagrangianMultiplier \
                               * math.pow((self.CurrentImplementableSolution.Production[w][t][p]), 2)
                               for p in self.Instance.ProductSet
                               for t in self.Instance.TimeBucketSet)

                qpconst += sum(0.5 * self.LagrangianMultiplier \
                               * math.pow((self.CurrentImplementableSolution.Consumption[w][t][c[0]][c[1]]), 2)
                               for c in self.Instance.ConsumptionSet
                               for t in self.Instance.TimeBucketSet)

                constant = lpconst + qpconst

                print("Python Cost of the scenario %r, penalties %r " % (self.CurrentSolution[w].TotalCost, penalty))
                print("Cost in cplex %r, ignored constant %r " % (self.MIPSolvers[w].Cplex.solution.get_objective_value(), constant))

                costwithconstant = self.MIPSolvers[w].Cplex.solution.get_objective_value() + constant
                actualcostwithpenalty = self.CurrentSolution[w].TotalCost + penalty
                print("cost with penalty from python %r / from cplex %r" % (actualcostwithpenalty, costwithconstant))

            #self.CurrentSolution[w].Print()


    def UpdateLagrangianCoeff(self, scenario):
        variables = []
        variablesquad = []
        mipsolver = self.MIPSolvers[scenario]
        for t in self.Instance.TimeBucketSet:
                for p in self.Instance.ProductSet:
                    variable = mipsolver.GetIndexQuantityVariable(p, t, 0)
                    coeff = mipsolver.GetQuantityCoeff(p, t, 0) + self.LagrangianQuantity[scenario][t][p]
                    variables.append((variable, coeff))
                    variablesquad.append((variable, variable, 0.5 * self.LagrangianMultiplier))

                    variable = mipsolver.GetIndexProductionVariable(p, t, 0)
                    coeff = mipsolver.GetProductionCefficient(p, t, 0) + self.LagrangianProduction[scenario][t][p]
                    variables.append((variable, coeff))
                    variablesquad.append((variable, variable, 0.5 * self.LagrangianMultiplier))

                for c in self.Instance.ConsumptionSet:
                        variable = mipsolver.GetIndexConsumptionVariable(c[1], c[0], t, 0)
                        coeff = mipsolver.GetConsumptionCoeff(c[1], c[0], t, 0) + self.LagrangianConsumption[scenario][t][c[1]][c[0]]
                        variables.append((variable, coeff))
                        variablesquad.append((variable, variable, 0.5 * self.LagrangianMultiplier))

        #print("New coeff: %r"%variables)
        mipsolver.Cplex.objective.set_linear(variables)
        #print("New quadratic coeff: %r" % variablesquad)

        mipsolver.Cplex.objective.set_quadratic_coefficients(variablesquad)

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
                sumprob = sum(self.ScenarioSet[w].Probability for w in scenarios)
                print("sumprob %r"%sumprob)
                # Average the quantities, and setups for this nodes.
                if time < self.Instance.NrTimeBucket:
                    for w in scenarios:
                        print("qty %r"%self.CurrentSolution[w].ProductionQuantity[0][time])

                    qty = [sum(self.ScenarioSet[w].Probability
                                * self.CurrentSolution[w].ProductionQuantity[0][time][p] for w in scenarios) \
                           / sumprob
                           for p in self.Instance.ProductSet]

                    prod = [sum(self.ScenarioSet[w].Probability
                                * self.CurrentSolution[w].Production[0][time][p] for w in scenarios)\
                           / sumprob
                             for p in self.Instance.ProductSet]

                    cons = [[sum(self.ScenarioSet[w].Probability
                                * self.CurrentSolution[w].Consumption[0][time][p][q] for w in scenarios) \
                           / sumprob
                             for p in self.Instance.ProductSet]
                            for q in self.Instance.ProductSet]

                    for w in scenarios:
                        for p in self.Instance.ProductSet:
                            solproduction[w][time][p] = prod[p]# int(round(prod[p]))
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
                    self.LinearLagQuantity[w][t][p], self.LagrangianQuantity[w][t][p] = \
                        self.ComputeLagrangian(self.LinearLagQuantity[w][t][p],
                                               self.CurrentSolution[w].ProductionQuantity[0][t][p],
                                               self.CurrentImplementableSolution.ProductionQuantity[w][t][p])

                    self.LinearLagProduction[w][t][p], self.LagrangianProduction[w][t][p] = \
                        self.ComputeLagrangian(self.LinearLagProduction[w][t][p],
                                               self.CurrentSolution[w].Production[0][t][p],
                                               self.CurrentImplementableSolution.Production[w][t][p])


                    for q in self.Instance.ProductSet:
                        self.LinearLagConsumption[w][t][p][q], self.LagrangianConsumption[w][t][p][q] = \
                            self.ComputeLagrangian(self.LinearLagConsumption[w][t][p][q],
                                                   self.CurrentSolution[w].Consumption[0][t][p][q],
                                                   self.CurrentImplementableSolution.Consumption[w][t][p][q])


    def ComputeLagrangian(self, prevlag, independentvalue, implementablevalue):
        linearlag = prevlag \
                     + (self.LagrangianMultiplier \
                         * (independentvalue \
                             - implementablevalue))
        lagrangian = linearlag \
                     - (self.LagrangianMultiplier \
                        * implementablevalue)
        return linearlag, lagrangian


    def ComputeConvergence(self):
        difference = 0
        for w in self.ScenarioNrSet:
            for t in self.Instance.TimeBucketSet:
                for p in self.Instance.ProductSet:
                    difference += self.ScenarioSet[w].Probability\
                                  * math.pow( self.CurrentSolution[w].ProductionQuantity[0][t][p] \
                                           - self.CurrentImplementableSolution.ProductionQuantity[w][t][p], 2)

                    difference += self.ScenarioSet[w].Probability\
                                  * math.pow(self.CurrentSolution[w].Production[0][t][p] \
                                           - self.CurrentImplementableSolution.Production[w][t][p], 2)


                    for q in self.Instance.ProductSet:
                        difference += self.ScenarioSet[w].Probability\
                                      * math.pow(self.CurrentSolution[w].Consumption[0][t][p][q] \
                                               - self.CurrentImplementableSolution.Consumption[w][t][p][q], 2)

        convergence = math.sqrt(difference)

        return convergence


    def PrintCurrentIteration(self):
        print("----------------Independent solutions--------------")
        for w in self.ScenarioNrSet:
            #self.CurrentSolution[w].Print()

            print("Scena %r: %r"%(w, self.CurrentSolution[w].ProductionQuantity))
        print("Implementable: %r" % ( self.CurrentImplementableSolution.ProductionQuantity))
        print("-----------IMPLEMENTABLE: -------------------------")
        self.CurrentImplementableSolution.Print()
        print("---------------------------------------------------")
        print("----------------------Multipliers------------------")
        print("Quantity:%r"%self.LagrangianQuantity)
        print("Linear Quantity:%r" % self.LinearLagQuantity)
        print("---------------------------------------------------")

    def WriteInTraceFile(self, string):
        if Constants.PrintSDDPTrace:
            self.TraceFile = open(self.TraceFileName, "a")
            self.TraceFile.write(string)
            self.TraceFile.close()

        # This function runs the SDDP algorithm

    def InitTrace(self):
        if Constants.PrintSDDPTrace:
            self.TraceFile = open(self.TraceFileName, "w")
            self.TraceFile.write("Start the Progressive Hedging algorithm \n")
            self.TraceFile.close()



    def GetDistance(self, solution):
        result = sum(self.ScenarioSet[w].Probability \
                     * math.pow(solution.ProductionQuantity[w][t][p], 2)
                     for p in self.Instance.ProductSet
                     for t in self.Instance.TimeBucketSet
                     for w in self.ScenarioNrSet)

        result += sum(self.ScenarioSet[w].Probability \
                      * math.pow(solution.Production[w][t][p], 2)
                      for p in self.Instance.ProductSet
                      for t in self.Instance.TimeBucketSet
                      for w in self.ScenarioNrSet)

        result += sum(self.ScenarioSet[w].Probability \
                      * math.pow(solution.Consumption[w][t][c[0]][c[1]], 2)
                      for c in self.Instance.ConsumptionSet
                      for t in self.Instance.TimeBucketSet
                      for w in self.ScenarioNrSet)

        return result

    def RateLargeChangeInImplementable(self):
        primalcon = self.GetPrimalConvergenceIndice()
        divider = max(self.GetDistance(self.CurrentImplementableSolution),
                      self.GetDistance(self.PreviousImplementableSolution))

        result =(primalcon / divider)
        return result

    def RatePrimalDual(self):
        primalcon = self.GetPrimalConvergenceIndice()
        dualcon = self.GetDualConvergenceIndice()
        divider = max(1,dualcon)

        result =(primalcon-dualcon / divider)
        return result

    def RateDualPrimal(self):
        primalcon = self.GetPrimalConvergenceIndice()
        dualcon = self.GetDualConvergenceIndice()
        divider = max(1, primalcon)

        result = (dualcon - primalcon / divider)
        return result

    def RateQuadLinear(self):
        result = self.GetQuadraticPenalty()/ (self.CurrentImplementableSolution.TotalCost + self.GetLinearPenalty())

        return result

    def GetPrimalConvergenceIndice(self):
        result = sum(self.ScenarioSet[w].Probability \
                     * math.pow(self.CurrentImplementableSolution.ProductionQuantity[w][t][p]
                                - self.PreviousImplementableSolution.ProductionQuantity[w][t][p], 2)
            for p in self.Instance.ProductSet
            for t in self.Instance.TimeBucketSet
            for w in self.ScenarioNrSet)

        result += sum(self.ScenarioSet[w].Probability \
                      * math.pow(self.CurrentImplementableSolution.Production[w][t][p]
                                 - self.PreviousImplementableSolution.Production[w][t][p], 2)
                    for p in self.Instance.ProductSet
                    for t in self.Instance.TimeBucketSet
                    for w in self.ScenarioNrSet)

        result += sum(self.ScenarioSet[w].Probability \
                      * math.pow(self.CurrentImplementableSolution.Consumption[w][t][c[0]][c[1]]
                                 - self.PreviousImplementableSolution.Consumption[w][t][c[0]][c[1]], 2)
                      for c in self.Instance.ConsumptionSet
                      for t in self.Instance.TimeBucketSet
                      for w in self.ScenarioNrSet)

        return result

    def GetQuadraticPenaltyConstant(self):
        result = sum(- self.ScenarioSet[w].Probability \
                     * self.LagrangianMultiplier * 0.5 \
                     * math.pow(self.CurrentImplementableSolution.ProductionQuantity[w][t][p], 2)
                     for p in self.Instance.ProductSet
                     for t in self.Instance.TimeBucketSet
                     for w in self.ScenarioNrSet)

        result += sum(- self.ScenarioSet[w].Probability \
                      * self.LagrangianMultiplier * 0.5 \
                      * math.pow(self.CurrentImplementableSolution.Production[w][t][p], 2)
                      for p in self.Instance.ProductSet
                      for t in self.Instance.TimeBucketSet
                      for w in self.ScenarioNrSet)

        result += sum(- self.ScenarioSet[w].Probability \
                      * self.LagrangianMultiplier * 0.5 \
                      * math.pow(self.CurrentImplementableSolution.Consumption[w][t][c[0]][c[1]], 2)
                      for c in self.Instance.ConsumptionSet
                      for t in self.Instance.TimeBucketSet
                      for w in self.ScenarioNrSet)

        return result

    def GetQuadraticPenalty(self):
        result = sum(self.ScenarioSet[w].Probability \
                     * self.LagrangianMultiplier \
                     * math.pow((self.CurrentSolution[w].ProductionQuantity[0][t][p]
                                 - self.CurrentImplementableSolution.ProductionQuantity[w][t][p]), 2)
                     for p in self.Instance.ProductSet
                     for t in self.Instance.TimeBucketSet
                     for w in self.ScenarioNrSet)

        result += sum(self.ScenarioSet[w].Probability \
                     * self.LagrangianMultiplier \
                     * math.pow((self.CurrentSolution[w].Production[0][t][p]
                                 - self.CurrentImplementableSolution.Production[w][t][p]), 2)
                      for p in self.Instance.ProductSet
                      for t in self.Instance.TimeBucketSet
                      for w in self.ScenarioNrSet)

        result += sum(self.ScenarioSet[w].Probability \
                     * self.LagrangianMultiplier \
                     * math.pow((self.CurrentSolution[w].Consumption[0][t][c[0]][c[1]]
                                - self.CurrentImplementableSolution.Consumption[w][t][c[0]][c[1]]), 2)
                      for c in self.Instance.ConsumptionSet
                      for t in self.Instance.TimeBucketSet
                      for w in self.ScenarioNrSet)

        return result

    def GetLinearPenaltyConstant(self):
        result = sum(-self.ScenarioSet[w].Probability \
                     * self.LinearLagQuantity[w][t][p] \
                     * (self.CurrentImplementableSolution.ProductionQuantity[w][t][p])
                     for p in self.Instance.ProductSet
                     for t in self.Instance.TimeBucketSet
                     for w in self.ScenarioNrSet)

        result += sum(-self.ScenarioSet[w].Probability \
                      * self.LinearLagProduction[w][t][p]
                      * (self.CurrentImplementableSolution.Production[w][t][p])
                      for p in self.Instance.ProductSet
                      for t in self.Instance.TimeBucketSet
                      for w in self.ScenarioNrSet)

        result += sum(-self.ScenarioSet[w].Probability \
                      * self.LinearLagConsumption[0][t][c[0]][c[1]] \
                      * (self.CurrentImplementableSolution.Consumption[w][t][c[0]][c[1]])
                      for c in self.Instance.ConsumptionSet
                      for t in self.Instance.TimeBucketSet
                      for w in self.ScenarioNrSet)

        return result

    def GetLinearPenalty(self):
        result = sum(self.ScenarioSet[w].Probability \
                     * self.LinearLagQuantity[w][t][p] \
                     * (self.CurrentSolution[w].ProductionQuantity[0][t][p]
                         - self.CurrentImplementableSolution.ProductionQuantity[w][t][p])
                     for p in self.Instance.ProductSet
                     for t in self.Instance.TimeBucketSet
                     for w in self.ScenarioNrSet)

        result += sum(self.ScenarioSet[w].Probability \
                      * self.LinearLagProduction[w][t][p]
                      * (self.CurrentSolution[w].Production[0][t][p]
                          - self.CurrentImplementableSolution.Production[w][t][p])
                     for p in self.Instance.ProductSet
                     for t in self.Instance.TimeBucketSet
                     for w in self.ScenarioNrSet)

        result += sum(self.ScenarioSet[w].Probability \
                      * self.LinearLagConsumption[0][t][c[0]][c[1]] \
                      * (self.CurrentSolution[w].Consumption[0][t][c[0]][c[1]]
                         - self.CurrentImplementableSolution.Consumption[w][t][c[0]][c[1]])
                     for c in self.Instance.ConsumptionSet
                     for t in self.Instance.TimeBucketSet
                     for w in self.ScenarioNrSet)

        return result

    def GetDualConvergenceIndice(self):
        result = sum(self.ScenarioSet[w].Probability \
                     * math.pow(self.CurrentSolution[w].ProductionQuantity[0][t][p]
                                  - self.CurrentImplementableSolution.ProductionQuantity[w][t][p],
                                  2)
                    for p in self.Instance.ProductSet
                    for t in self.Instance.TimeBucketSet
                    for w in self.ScenarioNrSet)

        result += sum(self.ScenarioSet[w].Probability \
                      * math.pow(self.CurrentSolution[w].Production[0][t][p]
                                  - self.CurrentImplementableSolution.Production[w][t][p],
                                  2)
                     for p in self.Instance.ProductSet
                     for t in self.Instance.TimeBucketSet
                     for w in self.ScenarioNrSet)

        result += sum(self.ScenarioSet[w].Probability \
                      * math.pow(self.CurrentSolution[w].Consumption[0][t][c[0]][c[1]]
                                  - self.CurrentImplementableSolution.Consumption[w][t][c[0]][c[1]],
                                  2)
                     for c in self.Instance.ConsumptionSet
                     for t in self.Instance.TimeBucketSet
                     for w in self.ScenarioNrSet)

        return result


    #This function run the algorithm
    def Run(self):
        self.InitTrace()
        self.CurrentSolution = [None for w in self.ScenarioNrSet]

        while not self.CheckStopingCriterion():

            # Solve each scenario independentely
            self.SolveScenariosIndependently()

            # Create an implementable solution on the scenario tree
            self.PreviousImplementableSolution = copy.deepcopy(self.CurrentImplementableSolution)
            self.CurrentImplementableSolution = self.CreateImplementableSolution()

            # Update the lagrangian multiplier
            self.UpdateLagragianMultipliers()

            self.CurrentIteration += 1

            if self.CurrentIteration == 5 :
                self.LagrangianMultiplier = 1

            #self.LagrangianMultiplier *= 0.9

            #if self.LagrangianMultiplier < 1:
            #    self.LagrangianMultiplier = 1

            if Constants.Debug:
                self.PrintCurrentIteration()

        return self.CurrentImplementableSolution