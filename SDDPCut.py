from __future__ import absolute_import, division, print_function
import cplex
from Constants import Constants
#from sets import Set

class SDDPCut(object):

    def __init__(self, owner=None, forwardstage=None, trial=-1):
        self.BackwarStage = owner
        self.ForwardStage = forwardstage
        self.BackwarStage.SDDPCuts.append(self)
        self.ForwardStage.SDDPCuts.append(self)
        self.Iteration = self.BackwarStage.SDDPOwner.CurrentIteration
        self.Trial = trial
        self.Id = len(self.BackwarStage.SDDPCuts) -1
        self.Name = "Cut_%d_%d" % (self.Iteration, self.Trial)
        self.Instance = self.BackwarStage.Instance

        self.CoefficientQuantityVariable = [[0 for p in self.Instance.ProductSet] for t in self.Instance.TimeBucketSet]
        self.CoefficientProductionVariable = [[0 for p in self.Instance.ProductSet] for t in self.Instance.TimeBucketSet]
        self.CoefficientStockVariable = [[0 for p in self.Instance.ProductSet] for t in self.Instance.TimeBucketSet]
        self.CoefficientBackorderyVariable = [[0 for p in self.Instance.ProductWithExternalDemand] for t in self.Instance.TimeBucketSet]

        #The quantity variable fixed at earlier stages with a non zero coefficient
        self.NonZeroFixedEarlierQuantityVar = set()
        self.NonZeroFixedEarlierProductionVar = set()
        self.NonZeroFixedEarlierStockVar = set()
        self.NonZeroFixedEarlierBackOrderVar = set()

        self.DemandRHS = 0.0
        self.CapacityRHS = 0.0
        self.PreviousCutRHS = 0.0
        self.InitialInventoryRHS = 0.0
        self.CPlexConstraint = None
        self.IsActive = False
        self.RHSConstantValue = -1
        self.RHSValueComputed = False

        #The index of the cut in the model
        self.IndexForward = []
        self.IndexBackward = []
        self.LastIterationWithDual = self.Iteration
        #self.LastAddedConstraintIndex = 0

        #This function add the cut to the MIP

    def Print(self):
        print("RHS:%s"%self.GetRHS())
        print("coefficients:")
        print("Quantities: %s"%self.CoefficientQuantityVariable)
        print("Production: %s" % self.CoefficientProductionVariable)
        print("Stock: %s" % self.CoefficientStockVariable)
        print("Backorder: %s" % self.CoefficientBackorderyVariable)



    #This function return the variables of the cut in its stage (do ot include the variable fixed at previous stage)
    def GetCutVariablesAtStage(self, stage, w):
        vars = [stage.GetIndexCostToGo(w)] \
               + [stage.GetIndexQuantityVariable(p, w) for p in self.Instance.ProductSet] \
               + [stage.GetIndexStockVariable(p, w) for p in stage.GetProductWithStockVariable()]
        if not self.BackwarStage.IsFirstStage():
            vars = vars + [stage.GetIndexBackorderVariable(p, w) for p in self.Instance.ProductWithExternalDemand]

        if self.BackwarStage.DecisionStage == 0:
            vars = vars + [stage.GetIndexProductionVariable(p, t) for p in self.Instance.ProductSet for t in self.Instance.TimeBucketSet]

        vars = vars + [stage.GetIndexCutRHSFromPreviousSatge(self)]

        return vars

    # This function return the coefficient variables of the cut in its stage (do ot include the variable fixed at previous stage)
    def GetCutVariablesCoefficientAtStage(self):
        coeff = [1] \
                + [self.CoefficientQuantityVariable[self.BackwarStage.GetTimePeriodAssociatedToQuantityVariable(p)][p] for p in
                   self.Instance.ProductSet] \
                + [self.CoefficientStockVariable[self.BackwarStage.GetTimePeriodAssociatedToInventoryVariable(p)][p] for p in
                   self.BackwarStage.GetProductWithStockVariable()]


        if not self.BackwarStage.IsFirstStage():
            coeff = coeff + [
                self.CoefficientBackorderyVariable[self.BackwarStage.GetTimePeriodAssociatedToBackorderVariable(p)][
                    self.Instance.ProductWithExternalDemandIndex[p]]
                for p in self.Instance.ProductWithExternalDemand ]

        if self.BackwarStage.DecisionStage == 0:
            coeff = coeff + [self.CoefficientProductionVariable[t][p] for p in self.Instance.ProductSet for t in
                             self.Instance.TimeBucketSet]

        coeff = coeff + [-1]
        return coeff

    def AddCut(self, addtomodel=True):
        self.IsActive = True
        if Constants.Debug:
            print("Add the Cut %s" %self.Name)

        #multiply by -1 because the variable goes on the left hand side
        righthandside = [self.ComputeCurrentRightHandSide()]

        if addtomodel:
            self.ActualyAddToModel(self.ForwardStage,   righthandside, True)
            if not self.BackwarStage.IsFirstStage() and self.BackwarStage.MIPDefined:
                self.ActualyAddToModel(self.BackwarStage,   righthandside, False)



    def ActualyAddToModel(self, stage,  righthandside, forward):

        RHSFromPreviousCuts = self.ComputeRHSFromPreviousStage(forward)
        stage.Cplex.variables.add(obj=[0.0],
                            lb=[RHSFromPreviousCuts],
                            ub=[RHSFromPreviousCuts])

        coeff = self.GetCutVariablesCoefficientAtStage()

        righthandside = [self.GetRHS()]
        for w in stage.FixedScenarioSet:
            vars = self.GetCutVariablesAtStage(stage, w)
            stage.Cplex.linear_constraints.add(lin_expr=[cplex.SparsePair(vars, coeff)],
                                                        senses=["G"],
                                                        rhs=righthandside)#,
                                                        #names =[self.Name])

            stage.IndexCutConstraint.append(stage.LastAddedConstraintIndex)

            if forward:
                self.IndexForward.append(stage.LastAddedConstraintIndex)
            else:
                self.IndexBackward.append(stage.LastAddedConstraintIndex)

            if Constants.Debug:
                stage.Cplex.linear_constraints.set_names(stage.LastAddedConstraintIndex, self.Name)

            stage.LastAddedConstraintIndex += 1

            stage.ConcernedCutinConstraint.append(self)

    def RemoveCut(self):
        self.RemoveCut(self.ForwardStage, True)
        self.RemoveCut(self.BackwarStage, False)

    def RemoveCut(self, stage, forward):
        self.IsActive = False
        print("Remove the Cut %s" % self.Name)
        if Constants.Debug:
            print("Remove the Cut %s" % self.Name)

        if forward:
            index = self.IndexForward
        else:
            index = self.IndexBackward

        nrcutremoved = len(index)
        for i in index:
            stage.Cplex.linear_constraints.delete(i)

        stage.LastAddedConstraintIndex -= nrcutremoved
        stage.IndexCutConstraint = self.Stage.IndexCutConstraint[0:-nrcutremoved]
        # renumber other cuts
        reindex = False
        for c in stage.ConcernedCutinConstraint:
            if reindex:
                c.Index = [i-nrcutremoved for i in c.Index]

            if c == self:
                reindex = True

        stage.ConcernedCutinConstraint.remove(self)
        if forward:
            self.IndexForward = []
        else:
            self.IndexBackward = []





    #This function modify the cut to take into account the Fixed variables
    def ModifyCut(self, forward):
        constrainttuples = []
        righthandside = self.ComputeCurrentRightHandSide()
        if forward:
            index = self.IndexForward
        else:
            index = self.IndexBackward

        for constrnr in index:
            constrainttuples.append((constrnr, righthandside))
        return constrainttuples



    def GetRHS(self):
        righthandside = self.RHSConstantValue
        return righthandside

    def ComputeCurrentRightHandSideA(self):

        righthandside = self.GetRHS()

        for p in self.Instance.ProductSet:
                for t in range(0,self.Stage.GetTimePeriodAssociatedToQuantityVariable(p)):
                   if self.CoefficientQuantityVariable[t][p] > 0:
                        righthandside = righthandside - self.Stage.SDDPOwner.GetQuantityFixedEarlier(p,t, self.Stage.CurrentTrialNr) \
                                                          * self.CoefficientQuantityVariable[t][p]

        if not self.Stage.IsFirstStage():
            for p in self.Instance.ProductSet:
                    for t in self.Instance.TimeBucketSet:
                        if self.CoefficientProductionVariable[t][p] > 0:
                            righthandside = righthandside - self.Stage.SDDPOwner.GetSetupFixedEarlier(p, t, self.Stage.CurrentTrialNr)\
                                                        * self.CoefficientProductionVariable[t][p]


        for p in self.Instance.ProductSet:
                for t in range(0, self.Stage.GetTimePeriodAssociatedToInventoryVariable(p)):
                    righthandside = righthandside - self.Stage.SDDPOwner.GetInventoryFixedEarlier(p, t, self.Stage.CurrentTrialNr) \
                                                    * self.CoefficientStockVariable[t][p]

        for p in self.Instance.ProductWithExternalDemand:
                for t in range(0, self.Stage.GetTimePeriodAssociatedToBackorderVariable(p)):

                    indexp = self.Instance.ProductWithExternalDemandIndex[p]
                    righthandside = righthandside - self.Stage.SDDPOwner.GetBackorderFixedEarlier(p, t, self.Stage.CurrentTrialNr) \
                                                    * self.CoefficientBackorderyVariable[t][indexp]


        return righthandside

    def ComputeCurrentRightHandSide(self):

        righthandside = self.GetRHS()


    def ComputeRHSFromPreviousStage(self, forward):
        if forward:
            scenarionr = self.ForwardStage.CurrentTrialNr
        else:
            scenarionr = self.BackwarStage.CurrentTrialNr

        result = 0
        for tuple in self.NonZeroFixedEarlierProductionVar:
            p = tuple[0]
            t = tuple[1]
            result = result - self.BackwarStage.SDDPOwner.GetSetupFixedEarlier(p, t, scenarionr) \
                                            * self.CoefficientProductionVariable[t][p]

        for tuple in self.NonZeroFixedEarlierQuantityVar:
            p = tuple[0]
            t = tuple[1]
            result = result - self.BackwarStage.SDDPOwner.GetQuantityFixedEarlier(p, t, scenarionr) \
                                            * self.CoefficientQuantityVariable[t][p]
        for tuple in self.NonZeroFixedEarlierBackOrderVar:
            p = tuple[0]
            t = tuple[1]
            indexp = self.Instance.ProductWithExternalDemandIndex[p]
            result = result - self.BackwarStage.SDDPOwner.GetBackorderFixedEarlier(p, t, scenarionr) \
                                            * self.CoefficientBackorderyVariable[t][indexp]

        for tuple in self.NonZeroFixedEarlierStockVar:
            p = tuple[0]
            t = tuple[1]
            result = result - self.BackwarStage.SDDPOwner.GetInventoryFixedEarlier(p, t, scenarionr) \
                                            * self.CoefficientStockVariable[t][p]
        return result

    #Increase the coefficient of the quantity variable for product and time  by value
    def IncreaseCoefficientQuantity(self, product, time, value):
        self.CoefficientQuantityVariable[time][product] = self.CoefficientQuantityVariable[time][product] + value

        if time < self.BackwarStage.GetTimePeriodAssociatedToQuantityVariable(product):
            self.NonZeroFixedEarlierQuantityVar.add((product, time))


    #Increase the coefficient of the quantity variable for product and time  by value
    def IncreaseCoefficientProduction(self, product, time, value):
        self.CoefficientProductionVariable[time][product] = self.CoefficientProductionVariable[time][product] + value

        if not self.BackwarStage.IsFirstStage():
            self.NonZeroFixedEarlierProductionVar.add((product, time))


        #Increase the coefficient of the quantity variable for product and time  by value
    def IncreaseCoefficientInventory(self, product, time, value):
        self.CoefficientStockVariable[time][product] = self.CoefficientStockVariable[time][product] + value
        if time < self.BackwarStage.GetTimePeriodAssociatedToInventoryVariable(product):
            self.NonZeroFixedEarlierStockVar.add((product, time))
    #Increase the coefficient of the quantity variable for product and time  by value
    def IncreaseCoefficientBackorder(self, product, time, value):
        indexp = self.Instance.ProductWithExternalDemandIndex[product]
        self.CoefficientBackorderyVariable[time][indexp] = self.CoefficientBackorderyVariable[time][indexp] + value

        if time < self.BackwarStage.GetTimePeriodAssociatedToBackorderVariable(product):
            self.NonZeroFixedEarlierBackOrderVar.add((product, time))

        # Increase the coefficient of the quantity variable for product and time  by value

    def IncreaseDemandRHS(self, value):
         self.DemandRHS = self.DemandRHS + value

    def IncreaseCapacityRHS(self, value):
        self.CapacityRHS = self.CapacityRHS + value

    def IncreasePReviousCutRHS(self, value):
        self.PreviousCutRHS = self.PreviousCutRHS + value

    def IncreaseInitInventryRHS(self, value):
        self.InitialInventoryRHS = self.InitialInventoryRHS + value

    def UpdateRHS (self):
        self.RHSConstantValue = self.DemandRHS + self.CapacityRHS + self.PreviousCutRHS + self.InitialInventoryRHS


    def GetCostToGoLBInCUrrentSolution(self,  w):
        variablofstage = self.GetCutVariablesAtStage(self.ForwardStage, 0)
        # REmove cost to go
        variablofstage = variablofstage[1:]
        # coefficient of the variable a

        valueofvariable = [self.ForwardStage.QuantityValues[w][p] for p in self.Instance.ProductSet] \
                          + [self.ForwardStage.InventoryValue[w][p] for p in self.ForwardStage.GetProductWithStockVariable()]

        if not self.ForwardStage.IsFirstStage():
            valueofvariable = valueofvariable + [
                self.ForwardStage.BackorderValue[w][self.Instance.ProductWithExternalDemandIndex[p]]
                for p in self.Instance.ProductWithExternalDemand]

        if self.BackwarStage.DecisionStage == 0:
            valueofvariable = valueofvariable + [self.ForwardStage.ProductionValue[w][t][p]
                                                 for p in self.Instance.ProductSet
                                                 for t in self.Instance.TimeBucketSet]

        coefficientvariableatstage = self.GetCutVariablesCoefficientAtStage()
        coefficientvariableatstage = coefficientvariableatstage[1:-1]
        valueofvarsinconsraint = sum(i[0] * i[1] for i in zip(valueofvariable, coefficientvariableatstage))

        RHS = self.ComputeRHSFromPreviousStage(False) + self.GetRHS()

        costtogo = RHS - valueofvarsinconsraint

        return costtogo

    def GetCostToGoLBInCorePoint(self, w):
        variablofstage = self.GetCutVariablesAtStage(self.ForwardStage, 0)
        #REmove cost to go
        variablofstage = variablofstage[1:]
        #coefficient of the variable a
        valueofvariable = [self.ForwardStage.CorePointQuantityValues[w][p] for p in self.Instance.ProductSet] \
                + [self.ForwardStage.CorePointInventoryValue[w][p] for p in self.ForwardStage.GetProductWithStockVariable()]

        if not self.ForwardStage.IsFirstStage():
             valueofvariable = valueofvariable + [self.ForwardStage.CorePointBackorderValue[w][self.Instance.ProductWithExternalDemandIndex[p]]
                                             for p in self.Instance.ProductWithExternalDemand]

        if self.ForwardStage.DecisionStage == 0:
            valueofvariable = valueofvariable + [self.ForwardStage.CorePointProductionValue[w][t][p]
                                                 for p in self.Instance.ProductSet
                                                 for t in self.Instance.TimeBucketSet]



        coefficientvariableatstage = self.GetCutVariablesCoefficientAtStage()
        coefficientvariableatstage = coefficientvariableatstage[1:-1]
        valueofvarsinconsraint = sum(i[0] * i[1] for i in zip(valueofvariable, coefficientvariableatstage))

        RHS = self.ComputeRHSFromPreviousStage(False) + self.GetRHS()

        costtogo = RHS - valueofvarsinconsraint
        return costtogo

