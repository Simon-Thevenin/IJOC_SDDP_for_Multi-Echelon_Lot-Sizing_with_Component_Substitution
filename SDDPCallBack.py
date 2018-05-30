import cplex
from Constants import Constants
from cplex.callbacks import LazyConstraintCallback
import copy
import time

class SDDPCallBack(LazyConstraintCallback):

    def __call__(self):
        if Constants.Debug:
            print("Enter in the call back")

        self.UpdateSolutionOfFirstStage()

        self.SDDPOwner.HeuristicSetupValue = [[self.Model.ProductionValue[0][t][p]
                                               for p in self.Model.Instance.ProductSet]
                                              for t in self.Model.Instance.TimeBucketSet]
        self.SDDPOwner.Stage[0].ChangeSetupToValueOfTwoStage()

        if Constants.PrintSDDPTrace:
            self.SDDPOwner.WriteInTraceFile("considered integer:%r \n"%self.Model.ProductionValue[0])

        UBequalLB = False

        while not UBequalLB:
            self.SDDPOwner.GenerateScenarios(self.SDDPOwner.CurrentNrScenario)
            self.SDDPOwner.ForwardPass(ignorefirststage=False)
            FirstStageCut, avgcostsubproblem = self.SDDPOwner.BackwardPass(returnfirststagecut=True)
            FirstStageCut.Stage = None
            FirstStageCutForModel = copy.deepcopy(FirstStageCut)
            FirstStageCut.Stage = self.SDDPOwner.Stage[0]
            FirstStageCutForModel.Stage = self.Model

            self.Model.CorePointQuantityValues = self.SDDPOwner.Stage[0].CorePointQuantityValues
            self.Model.CorePointProductionValue = self.SDDPOwner.Stage[0].CorePointProductionValue
            self.Model.CorePointInventoryValue = self.SDDPOwner.Stage[0].CorePointInventoryValue
            self.Model.CorePointBackorderValue = self.SDDPOwner.Stage[0].CorePointBackorderValue
            if Constants.Debug:
                self.Model.checknewcut(FirstStageCutForModel, avgcostsubproblem, self, withcorpoint=False)
            FirstStageCutForModel.AddCut(False)


            vars = FirstStageCutForModel.GetCutVariablesAtStage()
            coeff = FirstStageCutForModel.GetCutVariablesCoefficientAtStage()
            righthandside = [FirstStageCutForModel.ComputeCurrentRightHandSide()]

            if Constants.Debug:
                print("Add the constraint with var: %r"%vars)
                print(" coeff: %r" % coeff)
                print("rhs: %r" % righthandside)
               # vars = [0]
               # coeff = [1]
               # righthandside = [0.0]
            self.add(constraint=cplex.SparsePair(vars, coeff),
                     sense="G",
                     rhs=righthandside[0])

            if Constants.Debug:
                print("Constraint added")

            self.SDDPOwner.CurrentIteration += 1

        #self.write("model.lp")
            self.SDDPOwner.ComputeCost()
            self.SDDPOwner.UpdateLowerBound()
            self.SDDPOwner.UpdateUpperBound()
            if self.SDDPOwner.CurrentUpperBound < self.SDDPOwner.BestUpperBound:
                self.SDDPOwner.BestUpperBound = self.SDDPOwner.CurrentUpperBound
            UBequalLB = self.SDDPOwner.CheckStoppingRelaxationCriterion(100000) \
                        or self.SDDPOwner.CurrentLowerBound > self.SDDPOwner.BestUpperBound
            if Constants.PrintSDDPTrace:
                optimalitygap = (self.SDDPOwner.CurrentUpperBound - self.SDDPOwner.CurrentLowerBound)/self.SDDPOwner.CurrentUpperBound
                duration = time.time() - self.SDDPOwner.StartOfAlsorithm
                self.SDDPOwner.WriteInTraceFile("Iteration: %d, Duration: %d, LB: %r, UB: %r (exp:%r), Gap: %r \n" % (
                self.SDDPOwner.CurrentIteration, duration, self.SDDPOwner.CurrentLowerBound, self.SDDPOwner.CurrentUpperBound,
                self.SDDPOwner.CurrentExpvalueUpperBound, optimalitygap))

        if Constants.Debug:
            print("Exit call back")

    def UpdateSolutionOfFirstStage(self):
        self.Model.CurrentScenarioNr = 0
        sol = self
        self.Model.SaveSolutionFromSol(sol)
        self.Model.CopyDecisionOfScenario0ToAllScenario()
