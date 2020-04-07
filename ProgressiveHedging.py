# This class contains the attributes and methods allowing to define the progressive hedging algorithm.
from __future__ import division
from ScenarioTree import ScenarioTree
from Constants import Constants
from MIPSolver import MIPSolver
from Solution import Solution

import copy
import time
import math

class ProgressiveHedging(object):

    def __init__(self, instance, testidentifier, treestructure, scenariotree=None, givensetup=[], fixuntil=-2):


        self.Instance = instance

        self.TestIdentifier = testidentifier
        self.TreeStructure = treestructure

        self.GivenSetup = givensetup
        self.SolveWithFixedSetup = len(self.GivenSetup) > 0
        self.Evaluation = False



        self.GenerateScenarios(scenariotree)

        self.FixedUntil = fixuntil


        self.LagrangianMultiplier = 0.0
        self.CurrentImplementableSolution = None

        self.PreviouBeta = 0.5


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
        self.BuildMIPs2()

    #This function creates the scenario tree
    def GenerateScenarios(self, scenariotree=None):
        #Build the scenario tree
        if Constants.Debug:
            print(self.TreeStructure)
        if scenariotree is None:
            self.ScenarioTree = ScenarioTree(self.Instance, self.TreeStructure, self.TestIdentifier.ScenarioSeed,
                                             scenariogenerationmethod=self.TestIdentifier.ScenarioSampling,
                                             issymetric=Constants.MIPBasedOnSymetricTree,
                                             model=Constants.ModelYFix)
        else:
            self.ScenarioTree = scenariotree

        self.ScenarioSet = self.ScenarioTree.GetAllScenarios(False)
        #for s in self.ScenarioSet:
        #    print(s.Demands)
        self.ScenarioNrSet = range(len(self.ScenarioSet))
        self.SplitScenrioTree2()



    def BuildMIPs(self):
        #Build the mathematicals models (1 per scenarios)
        mipset = [0]
        #mipset = self.ScenarioNrSet
        self.MIPSolvers = [MIPSolver(self.Instance, Constants.ModelYFix, self.SplitedScenarioTree[w],
                                     implicitnonanticipativity=True, yfixheuristic=self.SolveWithFixedSetup,
                                     givensetups=self.GivenSetup, logfile="NO")
                           for w in mipset]

        self.SetFixedUntil(self.FixedUntil)

        for w in mipset:
            self.MIPSolvers[w].BuildModel()



    def BuildMIPs2(self):
        #Build the mathematicals models (1 per scenarios)
        mipset = [0]#range(self.NrMIPBatch)
        #mipset = self.ScenarioNrSet

        self.MIPSolvers = [MIPSolver(self.Instance, Constants.ModelYFix, self.SplitedScenarioTree[w], yfixheuristic=self.SolveWithFixedSetup,
                                     givensetups=self.GivenSetup, logfile="NO", expandfirstnode=True)
                           for w in mipset]

        self.SetFixedUntil(self.FixedUntil)

        for w in mipset:
            self.MIPSolvers[w].BuildModel()
        #    self.MIPSolvers[w].Cplex.write("test.lp")


    def SplitScenrioTree(self):

        treestructure = [1] + [1] * self.Instance.NrTimeBucket + [0]
        self.SplitedScenarioTree = [None for s in self.ScenarioNrSet]

        for scenarionr in self.ScenarioNrSet:
            scenario = self.ScenarioSet[scenarionr]
            self.SplitedScenarioTree[scenarionr] = ScenarioTree(self.Instance, treestructure, 0,
                                                              givenfirstperiod=scenario.Demands,
                                                              scenariogenerationmethod=self.TestIdentifier.ScenarioSampling,
                                                              model=Constants.ModelYFix)

            #justotest = self.SplitedScenarioTree[scenarionr].GetAllScenarios(False)
            #justotest[0].DisplayScenario()

    def SplitScenrioTree2(self):

        batchsize = 600
        self.NrMIPBatch = int(math.ceil(len(self.ScenarioNrSet)/(batchsize)))
        self.Indexscenarioinbatch = [None for m in range(self.NrMIPBatch)]
        self.Scenarioinbatch = [None for m in range(self.NrMIPBatch)]
        self.SplitedScenarioTree = [None for m in range(self.NrMIPBatch)]
        self.BatchofScenario = [int(math.floor(w/batchsize)) for w in self.ScenarioNrSet]
        self.NewIndexOfScenario = [ w % batchsize for w in self.ScenarioNrSet]


        for m in range(self.NrMIPBatch):

            firstscenarioinbatch = m * batchsize
            lastscenarioinbatch = min((m+1) * batchsize, len(self.ScenarioNrSet))
            nrscenarioinbatch = lastscenarioinbatch - firstscenarioinbatch
            self.Indexscenarioinbatch[m] = range(firstscenarioinbatch, lastscenarioinbatch)
            self.Scenarioinbatch[m] = [self.ScenarioSet[w] for w in self.Indexscenarioinbatch[m]]

            treestructure = [1] + [nrscenarioinbatch] + [1] * (self.Instance.NrTimeBucket-1) + [0]

            self.SplitedScenarioTree[m] = ScenarioTree(self.Instance, treestructure, 0,
                                                                    givenscenarioset=self.Scenarioinbatch[m],
                                                                    CopyscenariofromYFIX = True,
                                                                    scenariogenerationmethod=self.TestIdentifier.ScenarioSampling,
                                                                    model=Constants.ModelYFix)

                # justotest = self.SplitedScenarioTree[scenarionr].GetAllScenarios(False)
                # justotest[0].DisplayScenario()


    def CheckStopingCriterion(self):
        gap = Constants.Infinity

        if self.CurrentIteration > 0:
            gap = self.ComputeConvergenceY()

        convergencereached = gap < Constants.PHConvergenceTolerence

        duration = time.time() - self.StartTime
        timelimitreached = duration > Constants.AlgorithmTimeLimit
        iterationlimitreached = self.CurrentIteration > Constants.PHIterationLimit
        result = convergencereached or timelimitreached or iterationlimitreached

        if Constants.PrintSDDPTrace and self.CurrentIteration > 0:
            self.CurrentImplementableSolution.ComputeInventory()
            self.CurrentImplementableSolution.ComputeCost()

            dualconv = -1
            primconv = -1
            lpenalty = self.GetLinearPenalty()
            qpenalty = self.GetQuadraticPenalty()
            ratequad_lin = self.RateQuadLinear()
            ratechangeimplem = -1
            ratedualprimal = -1
            rateprimaldual = -1
            if self.CurrentIteration > 1:
                primconv = self.GetPrimalConvergenceIndice()
                dualconv = self.GetDualConvergenceIndice()
                ratechangeimplem = self.RateLargeChangeInImplementable()
                rateprimaldual = self.RatePrimalDual()
                ratedualprimal = self.RateDualPrimal()

            self.WriteInTraceFile("Iteration: %r Duration: %.2f Gap: %.5f UB:%.2f linear penalty:%.2f quadratic penalty: %.2f"
                                  " Multiplier:%.2f primal conv:%.2f dual conv:%.2f Rate Large Change(l): %.2f"
                                  " rate quad_lin(s):%.2f rateprimaldual(l<-):%.2f ratedualprimal(l->): %.2f  convergenceY:%r\n"
                                  % (self.CurrentIteration, duration, gap, self.CurrentImplementableSolution.TotalCost,
                                     lpenalty, qpenalty, self.LagrangianMultiplier, primconv, dualconv, ratechangeimplem,
                                     ratequad_lin, rateprimaldual, ratedualprimal, self.ComputeConvergenceY()))



        return result


    def SolveScenariosIndependently(self):
        #For each scenario
        for m in range(self.NrMIPBatch):#self.MIPSolvers:
       # for w in self.ScenarioNrSet:

            #Update the coeffient in the objective function
            self.UpdateLagrangianCoeff(m)
            #mip = self.MIPSolvers[w]
            mip = self.MIPSolvers[0]
            mip.ModifyMipForScenarioTree(self.SplitedScenarioTree[m])

            #Solve the model.
            #self.MIPSolvers[w].Cplex.write("moel.lp")

            self.CurrentSolution[m] = mip.Solve(True)

            #compute the cost for the penalty update strategy
            self.CurrentSolution[m].ComputeCost()

            if False and Constants.Debug and self.CurrentIteration > 1:

                qp = sum(math.pow((self.CurrentSolution[m].ProductionQuantity[0][t][p]
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

                qp = qp * self.LagrangianMultiplier * 0.5

                lp = sum(self.LinearLagQuantity[w][t][p] \
                          * (self.CurrentSolution[w].ProductionQuantity[0][t][p]
                             - self.CurrentImplementableSolution.ProductionQuantity[w][t][p])
                             for p in self.Instance.ProductSet
                             for t in self.Instance.TimeBucketSet)

                lp += sum(self.LinearLagProduction[w][t][p]
                              * (self.CurrentSolution[w].Production[0][t][p]
                                 - self.CurrentImplementableSolution.Production[w][t][p])
                              for p in self.Instance.ProductSet
                              for t in self.Instance.TimeBucketSet)

                lp += sum(self.LinearLagConsumption[w][t][c[0]][c[1]] \
                          * (self.CurrentSolution[w].Consumption[0][t][c[0]][c[1]]
                             - self.CurrentImplementableSolution.Consumption[w][t][c[0]][c[1]])
                              for c in self.Instance.ConsumptionSet
                              for t in self.Instance.TimeBucketSet)

                #print("lp %r + qp %r "%(lp, qp))
                penalty = lp + qp
                #penalty =  qp

                lpconst = sum(self.LinearLagQuantity[w][t][p] \
                              * (self.CurrentImplementableSolution.ProductionQuantity[w][t][p])
                              for p in self.Instance.ProductSet
                              for t in self.Instance.TimeBucketSet)

                lpconst += sum(self.LinearLagProduction[w][t][p]
                               * (self.CurrentImplementableSolution.Production[w][t][p])
                               for p in self.Instance.ProductSet
                               for t in self.Instance.TimeBucketSet)

                lpconst += sum(self.LinearLagConsumption[w][t][c[0]][c[1]] \
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

                #print("-lpconst %r + qpconst %r"%(lpconst, qpconst))
                constant = -lpconst + qpconst
                #constant =  qpconst

                #self.MIPSolvers[w].Cplex.solution.write("solCPLEX")

                #print("Python Cost of the scenario %r, penalties %r " % (self.CurrentSolution[w].TotalCost, penalty))
                #print("Cost in cplex %r, ignored constant %r " % (self.MIPSolvers[w].Cplex.solution.get_objective_value(), constant))

                costwithconstant = self.MIPSolvers[0].Cplex.solution.get_objective_value() + constant
                actualcostwithpenalty = self.CurrentSolution[w].TotalCost + penalty

                print("cost with penalty from python %r / from cplex %r" % (actualcostwithpenalty, costwithconstant))

                quadterm = self.GetQuadraticPenaltyForScenario(w)
                print("||||||||||||||||||||||quadterm term %r" % quadterm)
                linterm = self.GetLinearPenaltyForScenario(w)
                print("linear penalty term for scenario %r: %r" %(w,linterm))
                #self.CurrentSolution[w].Print()

    def GetQuadraticPenaltyForScenario(self, w):
        nw = self.NewIndexOfScenario[w]
        m = self.BatchofScenario[w]
        quadterm = sum(0.5 * self.LagrangianMultiplier \
                       * math.pow((self.CurrentSolution[m].ProductionQuantity[nw][t][p]), 2)
                       for p in self.Instance.ProductSet
                       for t in self.Instance.TimeBucketSet)

        quadterm += sum(0.5 * self.LagrangianMultiplier \
                        * math.pow((self.CurrentSolution[m].Production[nw][t][p]), 2)
                        for p in self.Instance.ProductSet
                        for t in self.Instance.TimeBucketSet)

        quadterm += sum(0.5 * self.LagrangianMultiplier \
                        * math.pow((self.CurrentSolution[m].Consumption[nw][t][c[0]][c[1]]), 2)
                        for c in self.Instance.ConsumptionSet
                        for t in self.Instance.TimeBucketSet)

        return quadterm

    def GetLinearPenaltyForScenario(self, w):

        nw =self.NewIndexOfScenario[w]
        m =self.BatchofScenario[w]
        linterm = sum(self.LagrangianQuantity[w][t][p] \
                      * (self.CurrentSolution[m].ProductionQuantity[nw][t][p])
                      for p in self.Instance.ProductSet
                      for t in self.Instance.TimeBucketSet)

        linterm += sum(self.LagrangianProduction[w][t][p] \
                       * (self.CurrentSolution[m].Production[nw][t][p])
                       for p in self.Instance.ProductSet
                       for t in self.Instance.TimeBucketSet)

        linterm += sum(self.LagrangianConsumption[w][t][c[0]][c[1]] \
                       * (self.CurrentSolution[m].Consumption[nw][t][c[0]][c[1]])
                       for c in self.Instance.ConsumptionSet
                       for t in self.Instance.TimeBucketSet)

        return linterm

    def UpdateLagrangianCoeff(self, batch):
        variables = []
        variablesquad = []
        mipsolver = self.MIPSolvers[0] #self.MIPSolvers[scenario]

        scenarioindexinmip = 0
        for scenario in self.Indexscenarioinbatch[batch]:
            for t in self.Instance.TimeBucketSet:
                    for p in self.Instance.ProductSet:
                        variable = mipsolver.GetIndexQuantityVariable(p, t, scenarioindexinmip)
                        coeff = mipsolver.GetQuantityCoeff(p, t, scenarioindexinmip) + self.LagrangianQuantity[scenario][t][p]
                        variables.append((variable, coeff))
                        variablesquad.append((variable, variable, 2 * 0.5 * self.LagrangianMultiplier))

                      #  variable = mipsolver.GetIndexProductionVariable(p, t, scenarioindexinmip)
                      #  coeff = mipsolver.GetProductionCefficient(p, t, scenarioindexinmip) + self.LagrangianProduction[scenario][t][p]
                      #  variables.append((variable, coeff))
                      #  variablesquad.append((variable, variable, 2 * 0.5 * self.LagrangianMultiplier))

                    for c in self.Instance.ConsumptionSet:
                            variable = int(mipsolver.GetIndexConsumptionVariable(c[0], c[1], t, scenarioindexinmip))
                            coeff = mipsolver.GetConsumptionCoeff(c[0], c[1], t, scenarioindexinmip) + self.LagrangianConsumption[scenario][t][c[0]][c[1]]
                            variables.append((variable, coeff))
                            variablesquad.append((variable, variable, 2 * 0.5 * self.LagrangianMultiplier))

            scenarioindexinmip += 1

        #print("New coeff: %r"%variables)
        mipsolver.Cplex.objective.set_linear(variables)
        #print("New quadratic coeff: %r" % variablesquad)

        mipsolver.Cplex.objective.set_quadratic_coefficients(variablesquad)
        if self.SolveWithFixedSetup:
            mipsolver.Cplex.set_problem_type(mipsolver.Cplex.problem_type.QP)
        else:
            mipsolver.Cplex.set_problem_type(mipsolver.Cplex.problem_type.MIQP)

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

        solbackorder = [[[-1 for p in self.Instance.ProductWithExternalDemand]
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
                # Average the quantities, and setups for this nodes.
                if time < self.Instance.NrTimeBucket:
                    #if Constants.Debug:
                       # for w in scenarios:

                       #     print("qty %r"%self.CurrentSolution[self.BatchofScenario[w]].ProductionQuantity[self.NewIndexOfScenario[w]][time])



                    qty = [round(sum(self.ScenarioSet[w].Probability
                                * self.CurrentSolution[self.BatchofScenario[w]].ProductionQuantity[self.NewIndexOfScenario[w]][time][p] for w in scenarios) \
                           / sumprob, 4)
                           for p in self.Instance.ProductSet]

                    inv = [round(sum(self.ScenarioSet[w].Probability
                                     * self.CurrentSolution[self.BatchofScenario[w]].InventoryLevel[self.NewIndexOfScenario[w]][time][p] for w in scenarios) \
                                 / sumprob, 4)
                           for p in self.Instance.ProductSet]

                    back = [round(sum(self.ScenarioSet[w].Probability
                                     * self.CurrentSolution[self.BatchofScenario[w]].BackOrder[self.NewIndexOfScenario[w]][time][self.Instance.ProductWithExternalDemandIndex[p]] for w in scenarios) \
                                 / sumprob, 4)
                           for p in self.Instance.ProductWithExternalDemand]

                    prod = [round(sum(self.ScenarioSet[w].Probability
                                * self.CurrentSolution[self.BatchofScenario[w]].Production[self.NewIndexOfScenario[w]][time][p] for w in scenarios)\
                           / sumprob,4)
                             for p in self.Instance.ProductSet]

                    cons = [[round(sum(self.ScenarioSet[w].Probability
                                        * self.CurrentSolution[self.BatchofScenario[w]].Consumption[self.NewIndexOfScenario[w]][time][q][p] for w in scenarios) \
                                     / sumprob,4)
                             for p in self.Instance.ProductSet]
                            for q in self.Instance.ProductSet]

                    for w in scenarios:
                        for p in self.Instance.ProductSet:
                            solproduction[w][time][p] = prod[p]# int(round(prod[p]))
                            solquantity[w][time][p] = qty[p]
                            solinventory[w][time][p] = inv[p]
                            if self.Instance.HasExternalDemand[p]:
                                indexp = self.Instance.ProductWithExternalDemandIndex[p]
                                solbackorder[w][time][indexp] = back[indexp]
                            for q in self.Instance.ProductSet:
                                solconsumption[w][time][q][p] = cons[q][p]


        solution = Solution(self.Instance, solquantity, solproduction, solinventory, solbackorder,
                            solconsumption,  self.ScenarioSet, self.ScenarioTree)


        return solution

    def UpdateLagragianMultipliers(self):
        for w in self.ScenarioNrSet:
            m = self.BatchofScenario[w]
            nw = self.NewIndexOfScenario[w]
            for t in self.Instance.TimeBucketSet:
                for p in self.Instance.ProductSet:
                    self.LinearLagQuantity[w][t][p], self.LagrangianQuantity[w][t][p] = \
                        self.ComputeLagrangian(self.LinearLagQuantity[w][t][p],
                                               self.CurrentSolution[m].ProductionQuantity[nw][t][p],
                                               self.CurrentImplementableSolution.ProductionQuantity[w][t][p])

                    self.LinearLagProduction[w][t][p], self.LagrangianProduction[w][t][p] = \
                        self.ComputeLagrangian(self.LinearLagProduction[w][t][p],
                                               self.CurrentSolution[m].Production[nw][t][p],
                                               self.CurrentImplementableSolution.Production[w][t][p])


                    for q in self.Instance.ProductSet:
                        self.LinearLagConsumption[w][t][p][q], self.LagrangianConsumption[w][t][p][q] = \
                            self.ComputeLagrangian(self.LinearLagConsumption[w][t][p][q],
                                                   self.CurrentSolution[m].Consumption[nw][t][p][q],
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

    def ComputeConvergenceY(self):
        difference = 0
        for w in self.ScenarioNrSet:
            nw = self.NewIndexOfScenario[w]
            m = self.BatchofScenario[w]
            for t in self.Instance.TimeBucketSet:
                for p in self.Instance.ProductSet:
                    difference += self.ScenarioSet[w].Probability \
                                  * math.pow(self.CurrentSolution[m].Production[nw][t][p] \
                                             - self.CurrentImplementableSolution.Production[w][t][p], 2)

        convergence = math.sqrt(difference)

        return convergence

    def ComputeConvergence(self):
        difference = 0
        for w in self.ScenarioNrSet:
            nw = self.NewIndexOfScenario[w]
            m = self.BatchofScenario[w]
            for t in self.Instance.TimeBucketSet:
                for p in self.Instance.ProductSet:
                    difference += self.ScenarioSet[w].Probability\
                                  * math.pow(self.CurrentSolution[m].ProductionQuantity[nw][t][p] \
                                           - self.CurrentImplementableSolution.ProductionQuantity[w][t][p], 2)

                    difference += self.ScenarioSet[w].Probability\
                                  * math.pow(self.CurrentSolution[m].Production[nw][t][p] \
                                           - self.CurrentImplementableSolution.Production[w][t][p], 2)

                    for q in self.Instance.ProductSet:
                        difference += self.ScenarioSet[w].Probability\
                                      * math.pow(self.CurrentSolution[m].Consumption[nw][t][p][q] \
                                               - self.CurrentImplementableSolution.Consumption[w][t][p][q], 2)

        convergence = math.sqrt(difference)

        return convergence


    def PrintCurrentIteration(self):
        print("----------------Independent solutions--------------")
        for w in self.ScenarioNrSet:
            #self.CurrentSolution[w].Print()
            print("Scena %r: %r"%(w, self.CurrentSolution[self.BatchofScenario[w]].ProductionQuantity[self.NewIndexOfScenario[w]]))

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


    def UpdateMultipler(self):


        teta = self.GetDualConvergenceIndice()
        delta = self.GetPrimalConvergenceIndice() + teta
        if self.CurrentIteration <= 2:
            tau = 1
        else:
            if self.PrevioudDualConvIndice == 0:
                self.PrevioudDualConvIndice = 0.000001
            tau = delta/self.PrevioudDualConvIndice

        self.PrevioudDualConvIndice = delta
        gamma = max(0.1, min(0.9, tau - 0.6))

        if self.CurrentIteration <= 2:
            sigma = 1
        else:
            sigma = (1-gamma) * self.PrevioudSigma + gamma * tau
        self.PrevioudSigma = sigma
        g = math.sqrt(1.1*sigma)

        if self.CurrentIteration <= 2:
            alpha = (teta / delta)
        else:
            if delta == 0:
                delta = 0.0001
            alpha = 0.8 * self.PreviouAlpha + 0.2 * (teta / delta)

        self.PreviouAlpha = alpha

        beta = 0.98 * self.PreviouBeta + 0.02 * alpha
        self.PreviouBeta = beta
        c = max(0.95, (1 - 2 * beta) / (1-beta))
        h = max(c + ((1-c) / beta) * alpha, 1 + (alpha - beta)/ (1-beta))
        q = math.pow(max(g, h), 1 / (1 + 0.01*(self.CurrentIteration - 2)))

        self.LagrangianMultiplier = max(0.01, min(100, self.LagrangianMultiplier * q))



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
        result = self.GetQuadraticPenalty()/ (self.GetLinearLagrangianterm())

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
                     * self.GetQuadraticPenaltyForScenario(w)
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

    def GetLinearLagrangianterm(self):


        result = sum( self.ScenarioSet[w].Probability * \
                      (self.GetLinearPenaltyForScenario(w) + self.CurrentSolution[self.BatchofScenario[w]].TotalCost)
                        for w in self.ScenarioNrSet)
        return result

    def GetLinearPenalty(self):
        result = sum(self.ScenarioSet[w].Probability * (self.GetLinearPenaltyForScenario(w))
                     for w in self.ScenarioNrSet)

        return result

    def GetDualConvergenceIndice(self):

        result = sum(self.ScenarioSet[w].Probability \
                     * math.pow(self.CurrentSolution[self.BatchofScenario[w]].ProductionQuantity[self.NewIndexOfScenario[w]][t][p]
                                  - self.CurrentImplementableSolution.ProductionQuantity[w][t][p],
                                  2)
                    for p in self.Instance.ProductSet
                    for t in self.Instance.TimeBucketSet
                    for w in self.ScenarioNrSet)



        result += sum(self.ScenarioSet[w].Probability \
                      * math.pow(self.CurrentSolution[self.BatchofScenario[w]].Production[self.NewIndexOfScenario[w]][t][p]
                                  - self.CurrentImplementableSolution.Production[w][t][p],
                                  2)
                     for p in self.Instance.ProductSet
                     for t in self.Instance.TimeBucketSet
                     for w in self.ScenarioNrSet)


        result += sum(self.ScenarioSet[w].Probability \
                      * math.pow(self.CurrentSolution[self.BatchofScenario[w]].Consumption[self.NewIndexOfScenario[w]][t][c[0]][c[1]]
                                  - self.CurrentImplementableSolution.Consumption[w][t][c[0]][c[1]],
                                  2)
                     for c in self.Instance.ConsumptionSet
                     for t in self.Instance.TimeBucketSet
                     for w in self.ScenarioNrSet)

        return result


    def UpdateForDemand(self, demanduptotimet):
        for w in self.ScenarioNrSet:
            for t in range(self.FixedUntil+1):
                for p in self.Instance.ProductSet:
                    self.ScenarioSet[w].Demands[t][p] = demanduptotimet[t][p]
        #    self.MIPSolvers[w].ModifyMipForScenario(demanduptotimet, self.FixedUntil+1)
        self.SplitScenrioTree()

    def UpdateForSetup(self, givensetup):
        for w in self.ScenarioNrSet:
            self.MIPSolvers[0].GivenSetup = givensetup
            self.MIPSolvers[0].UpdateSetup(givensetup)

    def UpdateForQuantity(self, givenquantity):
        for w in self.ScenarioNrSet:
            self.MIPSolvers[0].GivenQuantity = givenquantity
            self.MIPSolvers[0].ModifyMipForFixQuantity(givenquantity, self.FixedUntil+1)

    def UpdateForConsumption(self, givenconsumption):
        for w in self.ScenarioNrSet:
            self.MIPSolvers[0].GivenConsumption = givenconsumption
            self.MIPSolvers[0].ModifyMipForFixConsumption(givenconsumption, self.FixedUntil+1)

    def ReSetParameter(self):
        self.StartTime = time.time()
        self.LagrangianMultiplier = 0.0
        self.CurrentImplementableSolution = None

        self.PreviouBeta = 0.5

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

    def SetFixedUntil(self, time):
       # for w in self.ScenarioNrSet:
            self.MIPSolvers[0].FixSolutionUntil = time
            self.MIPSolvers[0].DemandKnownUntil = time + 1

    #This function run the algorithm
    def Run(self):

        self.PrintOnlyFirstStagePreviousValue = Constants.PrintOnlyFirstStageDecision
        if Constants.PrintOnlyFirstStageDecision:
            Constants.PrintOnlyFirstStageDecision = False
            # raise NameError("Progressive Hedging requires to print the full solution, set Constants.PrintOnlyFirstStageDecision to False")

        self.InitTrace()
        self.CurrentSolution = [None for w in self.ScenarioNrSet]

        while not self.CheckStopingCriterion():
            # Solve each scenario independentely
            self.SolveScenariosIndependently()

            # Create an implementable solution on the scenario tree
            self.PreviousImplementableSolution = copy.deepcopy(self.CurrentImplementableSolution)

            self.CurrentImplementableSolution = self.CreateImplementableSolution()

            self.CurrentIteration += 1

            if self.CurrentIteration == 1:
                    self.LagrangianMultiplier = 0.00001



            if False and self.CurrentIteration >= 2:
                self.UpdateMultipler()
            #self.LagrangianMultiplier *= 1.3
            #if self.LagrangianMultiplier < 1:
            #    self.LagrangianMultiplier = 1

            # Update the lagrangian multiplier
            self.UpdateLagragianMultipliers()

            #if Constants.Debug:
            #    self.PrintCurrentIteration()

        self.CurrentImplementableSolution.PHCost = self.CurrentImplementableSolution.TotalCost

        self.CurrentImplementableSolution.PHNrIteration = self.CurrentIteration
        self.CurrentImplementableSolution.ComputeInventory()
        self.CurrentImplementableSolution.ComputeCost()
        self.WriteInTraceFile("Enf of PH algorithm cost: %r"%self.CurrentImplementableSolution.TotalCost)

        Constants.PrintOnlyFirstStageDecision = self.PrintOnlyFirstStagePreviousValue

        return self.CurrentImplementableSolution