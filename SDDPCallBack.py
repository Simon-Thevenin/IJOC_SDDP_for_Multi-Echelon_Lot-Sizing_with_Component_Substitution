import cplex
from Constants import Constants
from cplex.callbacks import LazyConstraintCallback
#from cplex.callbacks import UserCutCallback
import copy
import time

class SDDPCallBack(LazyConstraintCallback):
#class SDDPCallBack(UserCutCallback):
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
        AddedCut = []
        AvgCostSubProb = []
        while not UBequalLB:
            #print("attention uncoment")
            print("Generate Trials in call back")
            self.SDDPOwner.GenerateTrialScenarios()
            self.SDDPOwner.ForwardPass(ignorefirststage=False)
            FirstStageCuts, avgsubprobcosts = self.SDDPOwner.BackwardPass(returnfirststagecut=True)
            print("End Backward pass")

            AddedCut = AddedCut + FirstStageCuts
            avgsubprobcosts = AvgCostSubProb + avgsubprobcosts

            self.SDDPOwner.CurrentIteration += 1

            self.SDDPOwner.ComputeCost()
            self.SDDPOwner.UpdateLowerBound()
            self.SDDPOwner.UpdateUpperBound()
            if self.SDDPOwner.CurrentUpperBound < self.SDDPOwner.BestUpperBound:
                self.SDDPOwner.BestUpperBound = self.SDDPOwner.CurrentUpperBound
            UBequalLB = self.SDDPOwner.CheckStoppingCriterion() #self.SDDPOwner.CheckStoppingRelaxationCriterion(100000) # \
                       # or self.SDDPOwner.CurrentLowerBound > self.SDDPOwner.BestUpperBound
            #if Constants.PrintSDDPTrace:
            #    optimalitygap = (self.SDDPOwner.CurrentUpperBound - self.SDDPOwner.CurrentLowerBound)
            #    duration = time.time() - self.SDDPOwner.StartOfAlsorithm
            #    self.SDDPOwner.WriteInTraceFile("Iteration: %d, Duration: %d, LB: %r, UB: %r (exp:%r), Gap: %r \n" % (
            #    self.SDDPOwner.CurrentIteration, duration, self.SDDPOwner.CurrentLowerBound, self.SDDPOwner.CurrentUpperBound,
            #    self.SDDPOwner.CurrentExpvalueUpperBound, optimalitygap))

        print("Actually add the cuts")
        print ("added cuts %r"%AddedCut)
        addedcutindex = [cut.Index for cut in AddedCut]
        self.SDDPOwner.Stage[0].Cplex.solve()
        sol = self.SDDPOwner.Stage[0].Cplex.solution
        AddedCutDuals = sol.get_dual_values(addedcutindex)
       # sol.write("./Temp/solinlazycut_%s.sol" % (self.SDDPOwner.CurrentIteration))
        print AddedCut
        for c in range(len(AddedCut)):
            cut = AddedCut[c]
            #avgcostsubproblem = avgsubprobcosts[c]
            dual = AddedCutDuals[c]
            print("%s:%s,%s" % (cut.Name, dual, addedcutindex[c]))

            if False and dual == 0:
                cut.RemoveCut()
            else:
                cut.Stage = None
                FirstStageCutForModel = copy.deepcopy(cut)
                cut.Stage = self.SDDPOwner.Stage[0]
                FirstStageCutForModel.Stage = self.Model

                self.Model.CorePointQuantityValues = self.SDDPOwner.Stage[0].CorePointQuantityValues
                self.Model.CorePointProductionValue = self.SDDPOwner.Stage[0].CorePointProductionValue
                self.Model.CorePointInventoryValue = self.SDDPOwner.Stage[0].CorePointInventoryValue
                self.Model.CorePointBackorderValue = self.SDDPOwner.Stage[0].CorePointBackorderValue
                if Constants.Debug:
                    print("THERE IS NO CHECK!!!!!!!!!!!!!!")
                    #self.Model.checknewcut(FirstStageCutForModel, avgcostsubproblem, self, None, withcorpoint=False)
                FirstStageCutForModel.AddCut(False)

                vars = FirstStageCutForModel.GetCutVariablesAtStage()
                coeff = FirstStageCutForModel.GetCutVariablesCoefficientAtStage()
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

        self.Model.Cplex.write("./Temp/yyoyoyo.lp")

        if Constants.Debug:
            print("Exit call back")

    def UpdateSolutionOfFirstStage(self):
        self.Model.CurrentScenarioNr = 0
        sol = self
        self.Model.SaveSolutionFromSol(sol)
        self.Model.CopyDecisionOfScenario0ToAllScenario()
