import cplex
from Constants import Constants
from cplex.callbacks import LazyConstraintCallback
#from cplex.callbacks import UserCutCallback
import copy
import time
from ScenarioTree import ScenarioTree
from MIPSolver import MIPSolver
import random

class SDDPCallBack(LazyConstraintCallback):
#class SDDPCallBack(UserCutCallback):
    def __call__(self):
        if Constants.Debug:
            print("Enter in the call back")
        self.SDDPOwner.NrIterationWithoutLBImprovment = 0
        self.LastIterationWithTest = self.SDDPOwner.CurrentIteration
        self.UpdateSolutionOfFirstStage()

        newsetup = [[self.Model.ProductionValue[0][t][p]
                      for p in self.Model.Instance.ProductSet]
                     for t in self.Model.Instance.TimeBucketSet]

        samesetupasprevious = (newsetup == self.SDDPOwner.HeuristicSetupValue) and Constants.SDDPFixSetupStrategy

        self.SDDPOwner.HeuristicSetupValue = [[self.Model.ProductionValue[0][t][p]
                                               for p in self.Model.Instance.ProductSet]
                                              for t in self.Model.Instance.TimeBucketSet]
        self.SDDPOwner.ForwardStage[0].ChangeSetupToValueOfTwoStage(makecontinuous=True)

        if Constants.PrintSDDPTrace:
            self.SDDPOwner.WriteInTraceFile("considered integer:%r \n"%self.Model.ProductionValue[0])
            self.SDDPOwner.WriteInTraceFile("considered quantity:%r \n"%self.Model.QuantityValues[0])
            self.SDDPOwner.WriteInTraceFile("Same setup as previous %r \n"%samesetupasprevious)
            self.SDDPOwner.WriteInTraceFile("Current Cost in B&B %r \n" % self.get_objective_value())
        ShouldStop = False
        firstiteration =True
        AddedCut = []
        AvgCostSubProb = []
        while not ShouldStop:
            #print("attention uncoment")


            if firstiteration and Constants.SDDPFirstForwardWithEVPI:
                solution = self.SolveForwardAsEVPI()
            else:
                self.SDDPOwner.CurrentForwardSampleSize = 1
                self.SDDPOwner.GenerateTrialScenarios()
                self.SDDPOwner.ForwardPass(ignorefirststage=False)
            FirstStageCuts, avgsubprobcosts = self.SDDPOwner.BackwardPass(returnfirststagecut=True)
            AddedCut = AddedCut + FirstStageCuts
            avgsubprobcosts = AvgCostSubProb + avgsubprobcosts

            self.SDDPOwner.CurrentIteration += 1
            self.SDDPOwner.ComputeCost()
            self.SDDPOwner.UpdateLowerBound()
            self.SDDPOwner.UpdateUpperBound()

            if not firstiteration or not Constants.SDDPFirstForwardWithEVPI:
                UBequalLB = self.SDDPOwner.CheckStoppingCriterion()
            else:
                UBequalLB = Constants.SDDPFirstForwardWithEVPI
                self.SDDPOwner.WriteInTraceFile(
                    "Iteration With EVPI Forward: %d,  LB: %r, (exp UB:%r) \n"
                    % (self.SDDPOwner.CurrentIteration, self.SDDPOwner.CurrentLowerBound, solution.TotalCost))

            if self.SDDPOwner.LastExpectedCostComputedOnAllScenario < self.SDDPOwner.BestUpperBound:
                self.SDDPOwner.BestUpperBound = self.SDDPOwner.LastExpectedCostComputedOnAllScenario
                self.SDDPOwner.CurrentBestSetups = self.SDDPOwner.HeuristicSetupValue
                self.SDDPOwner.WriteInTraceFile("New Best Upper Bound!!!! :%r \n" % self.SDDPOwner.BestUpperBound)
             #self.SDDPOwner.CheckStoppingRelaxationCriterion(100000) # \
                        # or self.SDDPOwner.CurrentLowerBound > self.SDDPOwner.BestUpperBound

            ShouldStop = UBequalLB or (self.SDDPOwner.CurrentLowerBound > self.SDDPOwner.BestUpperBound and not samesetupasprevious)
            firstiteration = False
            #if Constants.PrintSDDPTrace:
            #    optimalitygap = (self.SDDPOwner.CurrentUpperBound - self.SDDPOwner.CurrentLowerBound)
            #    duration = time.time() - self.SDDPOwner.StartOfAlsorithm
            #    self.SDDPOwner.WriteInTraceFile("Iteration: %d, Duration: %d, LB: %r, UB: %r (exp:%r), Gap: %r \n" % (
            #    self.SDDPOwner.CurrentIteration, duration, self.SDDPOwner.CurrentLowerBound, self.SDDPOwner.CurrentUpperBound,
            #    self.SDDPOwner.CurrentExpvalueUpperBound, optimalitygap))
        if Constants.Debug:
            print("Actually add the cuts")
            print ("added cuts %r"%AddedCut)
        addedcutindex = [cut.Id for cut in AddedCut]
        #self.SDDPOwner.ForwardStage[0].Cplex.solve()
        #sol = self.SDDPOwner.ForwardStage[0].Cplex.solution
        #AddedCutDuals = sol.get_dual_values(addedcutindex)
       # sol.write("./Temp/solinlazycut_%s.sol" % (self.SDDPOwner.CurrentIteration))

        for c in range(len(AddedCut)):
            cut = AddedCut[c]
            #avgcostsubproblem = avgsubprobcosts[c]
         #   dual = AddedCutDuals[c]

          #  if False and dual == 0:
           #     cut.RemoveCut()
            #else:
            cut.ForwardStage = None
            backcut = cut.BackwarStage
            cut.BackwarStage = None
            FirstStageCutForModel = copy.deepcopy(cut)
            cut.ForwardStage = self.SDDPOwner.ForwardStage[0]
            cut.BackwarStage = self.SDDPOwner.ForwardStage[0]
            FirstStageCutForModel.ForwardStage = self.Model
            FirstStageCutForModel.BackwarStage = self.Model#self.SDDPOwner.BackwardStage[0]

            self.Model.CorePointQuantityValues = self.SDDPOwner.ForwardStage[0].CorePointQuantityValues
            self.Model.CorePointProductionValue = self.SDDPOwner.ForwardStage[0].CorePointProductionValue
            self.Model.CorePointInventoryValue = self.SDDPOwner.ForwardStage[0].CorePointInventoryValue
            self.Model.CorePointBackorderValue = self.SDDPOwner.ForwardStage[0].CorePointBackorderValue
            if Constants.Debug:
                    print("THERE IS NO CHECK!!!!!!!!!!!!!!")
                    #self.Model.checknewcut(FirstStageCutForModel, avgcostsubproblem, self, None, withcorpoint=False)
            FirstStageCutForModel.AddCut(False)
            vars = FirstStageCutForModel.GetCutVariablesAtStage(self.Model, 0)
            vars = vars[0:-1]
            coeff = FirstStageCutForModel.GetCutVariablesCoefficientAtStage()
            coeff = coeff[0:-1]
            righthandside = [FirstStageCutForModel.ComputeCurrentRightHandSide()]

            if Constants.Debug:
                    print("Add the constraint with var: %r" % vars)
                    print(" coeff: %r" % coeff)
                    print("rhs: %r" % righthandside)
                # vars = [0]
                # coeff = [1]
                # righthandside = [0.0]
            self.add(constraint=cplex.SparsePair(vars, coeff),
                         # cut=cplex.SparsePair(vars, coeff),
                         sense="G",
                         rhs=righthandside[0])

            if Constants.Debug:
                print("Constraint added")
        if Constants.Debug:
             self.Model.Cplex.write("./Temp/yyoyoyo.lp")

        self.SDDPOwner.IsIterationWithConvergenceTest = False
        self.SDDPOwner.GenerateTrialScenarios()

        if Constants.Debug:
            print("Exit call back")

    def UpdateSolutionOfFirstStage(self):
        self.Model.CurrentTrialNr = 0

        sol = self
        self.Model.SaveSolutionFromSol(sol)
        self.Model.CopyDecisionOfScenario0ToAllScenario()




    def SolveForwardAsEVPI(self):
        if Constants.Debug:
            print("Build the MIP with fix setups and single scenario")

        #self.SDDPOwner.GenerateTrialScenarios()
        #scenario = self.SDDPOwner.CurrentSetOfTrialScenarios
        #treestructure = [1, len(scenario)] + [1] * (self.SDDPOwner.Instance.NrTimeBucket - 1) + [0]


        #scenariotree = ScenarioTree(self.SDDPOwner.Instance, treestructure, 0,
        #                            CopyscenariofromYFIX=True,
        #                            givenscenarioset=scenario)

        givenquantity = self.Model.QuantityValues[0]

        if  not self.Model.FullTreeSolverDefine:
            self.Model.FullTreeSolver = MIPSolver(self.SDDPOwner.Instance, Constants.ModelYFix,
                                              self.SDDPOwner.SAAScenarioTree, yfixheuristic=True,
                                              implicitnonanticipativity=True,
                                              givensetups=self.SDDPOwner.HeuristicSetupValue,
                                              givenquantities= givenquantity,
                                              givenconsumption= self.Model.ConsumptionValues[0],
                                              evaluatesolution=True,
                                              fixsolutionuntil=self.Model.TimeDecisionStage + len(self.Model.RangePeriodQty) -1)

            self.Model.FullTreeSolver.BuildModel()
            self.Model.FullTreeSolverDefine = True
        else:
            self.Model.FullTreeSolver.ModifyMIPForSetup(self.SDDPOwner.HeuristicSetupValue)
            self.Model.FullTreeSolver.ModifyMipForFixConsumption(self.Model.ConsumptionValues[0], self.Model.TimeDecisionStage + len(self.Model.RangePeriodQty)  )
            self.Model.FullTreeSolver.ModifyMipForFixQuantity(self.Model.QuantityValues[0], self.Model.TimeDecisionStage + len(self.Model.RangePeriodQty) )

        if Constants.Debug:
            print("Start to solve instance %s with Cplex" % self.SDDPOwner.Instance.InstanceName)

        solution = self.Model.FullTreeSolver.Solve()


        if Constants.Debug:
            print("copy the decision in the right table")

        self.SDDPOwner.CurrentNrScenario = len(self.SDDPOwner.CompleteSetOfSAAScenario)

        self.SDDPOwner.CurrentSetOfTrialScenarios = []
        scenarionr = []
        for w in range(self.SDDPOwner.CurrentNrScenario):
             # random.randint(0, len(self.SDDPOwner.CompleteSetOfSAAScenario)-1)
             scenarionr.append(w)

             selected = self.SDDPOwner.CompleteSetOfSAAScenario[w]
             self.SDDPOwner.CurrentSetOfTrialScenarios.append(selected)


        self.SDDPOwner.TrialScenarioNrSet = range(len(self.SDDPOwner.CurrentSetOfTrialScenarios))
        self.SDDPOwner.CurrentNrScenario = len(self.SDDPOwner.CurrentSetOfTrialScenarios)
        self.SDDPOwner.SDDPNrScenarioTest = self.SDDPOwner.CurrentNrScenario
        for stage in self.SDDPOwner.StagesSet:
            self.SDDPOwner.ForwardStage[stage].SetNrTrialScenario(self.SDDPOwner.CurrentNrScenario)
            self.SDDPOwner.ForwardStage[stage].FixedScenarioPobability = [1]
            self.SDDPOwner.BackwardStage[stage].SAAStageCostPerScenarioWithoutCostoGopertrial = [0 for w in
                                                                                       self.SDDPOwner.TrialScenarioNrSet]

        stages = self.SDDPOwner.ForwardStage + [self.Model]
        # print("Scenario is 0, because this should only be used in forward pass")
        for i in self.SDDPOwner.TrialScenarioNrSet:
            w = scenarionr[i]

            for stage in stages:

                if len(stage.RangePeriodQty) > 0:
                    stage.QuantityValues[i] = [[solution.ProductionQuantity[w][stage.PeriodsInGlobalMIPQty[t]][p]
                                                                 for p in stage.Instance.ProductSet]
                                                                for t in stage.RangePeriodQty]

                    stage.ConsumptionValues[i] = [[solution.Consumption[w][stage.PeriodsInGlobalMIPQty[t]][c[0]][c[1]]
                                                   for c in stage.Instance.ConsumptionSet]
                                                   for t in stage.RangePeriodQty]


                if stage.IsFirstStage():
                    stage.ProductionValue[i] = [[solution.Production[w][t][p]
                                                                  for p in stage.Instance.ProductSet]
                                                                 for t in stage.Instance.TimeBucketSet]

                stage.InventoryValue[i] = [['nan' for p in stage.Instance.ProductSet]
                                                   for t in stage.RangePeriodInv]

                stage.BackorderValue[i] = [['nan' for p in stage.Instance.ProductSet]
                                                             for t in stage.RangePeriodInv]


                for t in stage.RangePeriodInv:
                    for p in stage.GetProductWithStockVariable(t):
                        if stage.Instance.HasExternalDemand[p]:
                            indexp = self.SDDPOwner.Instance.ProductWithExternalDemandIndex[p]
                            time = stage.GetTimePeriodAssociatedToInventoryVariable(p,t)
                            stage.InventoryValue[i][t][p] = solution.InventoryLevel[w][time][p]
                            stage.BackorderValue[i][t][p] = solution.BackOrder[w][time][indexp]

                        else:
                            time = stage.GetTimePeriodAssociatedToInventoryVariable( p,t)


                            stage.InventoryValue[i][t][p] = solution.InventoryLevel[w][time][p]

        if Constants.Debug:
            print("out of solve LP with tree")
        return solution

            # for stage in self.SDDPOwner.ForwardStage:
            #     print(stage.QuantityValues[0])
            #
            # for stage in self.SDDPOwner.ForwardStage:
            #     print(stage.ConsumptionValues[0])
            #
            # for stage in self.SDDPOwner.ForwardStage:
            #     print(stage.ProductionValue[0])
            #
            # for stage in self.SDDPOwner.ForwardStage:
            #     print(stage.InventoryValue[0])
            #
            # for stage in self.SDDPOwner.ForwardStage:
            #     print(stage.BackorderValue[0])

