#This class contains the code to implement SDDP in a single search tree. The results were negative (it did not improve the performance), and they are not included in the paper.
import cplex
from Constants import Constants
from cplex.callbacks import LazyConstraintCallback
#from cplex.callbacks import UserCutCallback
import copy
import time
from ScenarioTree import ScenarioTree
from MIPSolver import MIPSolver
import random

class CallBackML(LazyConstraintCallback):
#class SDDPCallBack(UserCutCallback):
    def __call__(self):
        if Constants.Debug:
             print "enter call back"
        indexarray = [self.SDDPOwner.ForwardStage[0].GetIndexProductionVariable(p, t) for t in
                      self.SDDPOwner.Instance.TimeBucketSet
                      for p in self.SDDPOwner.Instance.ProductSet]
        values = self.get_values(indexarray)

        self.MLLocalSearch.GivenSetup2D =  [[max(values[t * self.SDDPOwner.Instance.NrProduct + p], 0.0)
                       for p in self.SDDPOwner.Instance.ProductSet]
                      for t in self.SDDPOwner.Instance.TimeBucketSet]
        if Constants.Debug:
            print "run sddp"

        solution = self.MLLocalSearch.RunSDDP()

        self.MLLocalSearch.updateRecord(solution)

        AddedCut = self.MLLocalSearch.FirstStageCutAddedInLastSDDP

        for c in range(len(AddedCut)):
            cut = AddedCut[c]

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
                    #Uncomment the line below to check if the added cut is valid
                    #self.Model.checknewcut(FirstStageCutForModel, avgcostsubproblem, self, None, withcorpoint=False)
            FirstStageCutForModel.AddCut(False)

            coeff = cut.GetCutVariablesCoefficientAtStage()

            righthandside = cut.GetRHS()




            for w in self.SDDPOwner.ForwardStage[0].FixedScenarioSet:

                vars = cut.GetCutVariablesAtStage(self.SDDPOwner.ForwardStage[0], w)
                vars = vars[0:-1]
                coeffs = [1.0] + coeff[0:-1]


                self.add(cplex.SparsePair(vars, coeffs),
                         sense="G",
                         rhs=righthandside)


            if Constants.Debug:
                self.Model.Cplex.write("./Temp/ModelCreatedByCallBackML.lp")