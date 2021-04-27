from __future__ import absolute_import, division, print_function
import cplex
import math
from Constants import Constants
from SDDPStage import SDDPStage
import numpy as np

# This class contains the attributes and methodss allowing to define one stage of the SDDP algorithm.
class SDDPLastStage( SDDPStage ):
    def rescaletimestock(self, t, p):

        result = t - self.GetStartStageTimeRangeStock(p)
        return result

    def rescaletimequantity(self, t):
        result = t - (self.Instance.NrTimeBucket - self.Instance.NrTimeBucketWithoutUncertaintyAfter)
        return result

    def GetLastStageTimeRangeQuantity(self):
       result = range(self.GetStartStageTimeRangeQuantity(), self.Instance.NrTimeBucket)
       return result

    #return the number of time unit at the end of horizon without stochasticity
    def GetLastStageTimeRangeStock(self, p):
        if self.Instance.HasExternalDemand[p]:
            result = range(self.Instance.NrTimeBucket - self.Instance.NrTimeBucketWithoutUncertaintyAfter -1, self.Instance.NrTimeBucket)
        else:
            result = range(self.Instance.NrTimeBucket - self.Instance.NrTimeBucketWithoutUncertaintyAfter,
                           self.Instance.NrTimeBucket)
        return result

    def GetNrStageStock(self, p):
        if self.Instance.HasExternalDemand[p]:
            result = self.Instance.NrTimeBucketWithoutUncertaintyAfter +1
        else:
            result = self.Instance.NrTimeBucketWithoutUncertaintyAfter
        return result

    def GetNrStageQuantity(self):
        result = self.Instance.NrTimeBucketWithoutUncertaintyAfter + 1
        return result

    #The last stage has multiple inventory variable (for all time period without uncertainty) return the earliest time period with invenotry variabele
    def GetStartStageTimeRangeStock(self, p):
        if self.Instance.HasExternalDemand[p]:
            result = (self.Instance.NrTimeBucket - self.Instance.NrTimeBucketWithoutUncertaintyAfter - 1)
        else:
            result = (self.Instance.NrTimeBucket - self.Instance.NrTimeBucketWithoutUncertaintyAfter)
        return result


    def GetStartStageTimeRangeQuantity(self):
        result = self.Instance.NrTimeBucket - self.Instance.NrTimeBucketWithoutUncertaintyAfter -1
        return result

    #Compute the number of variable of each type (production quantty, setup, inventory, backorder)
    def ComputeNrVariables2(self):
        #number of variable at stage 1<t<T
        self.NrQuantityVariable = 0#self.Instance.NrProduct * (self.Instance.NrTimeBucketWithoutUncertainty)
        self.NrStockVariable = len(self.FixedScenarioSet)*sum(self.GetNrStageStock(q) for q in self.Instance.ProductSet)
        self.NrBackOrderVariable = len(self.FixedScenarioSet)*sum(self.GetNrStageStock(q) for q in self.Instance.ProductWithExternalDemand)
        self.NrStockVariablePerScenario = sum(self.GetNrStageStock(q) for q in self.Instance.ProductSet)
        self.NrBackOrderVariablePerScenario = sum(self.GetNrStageStock(q) for q in self.Instance.ProductWithExternalDemand)
        self.NRFlowFromPreviousStage = sum(self.GetNrStageStock(q) for q in self.Instance.ProductSet)
        self.NrProductionVariable = 0


    def GetIndexBackorderVariable2(self, p,  w):
        indexp = self.Instance.ProductWithExternalDemandIndex[p]
        return self.StartBackOrder + w * self.NrBackOrderVariablePerScenario + indexp

    # Return the index of the variable associated with the quanity of product p decided at the current stage
    def GetIndexQuantityVariable2(self, p, w):
        return self.StartQuantity + w*self.NrQuantityVariable + p * self.Instance.NrTimeBucketWithoutUncertaintyAfter + p #self.rescaletimequantity(t)

    def GetIndexStockVariable2(self, p, w):
        #result = sum(self.GetNrStageStock(q) for q in range(0, p))
        indexp = self.Instance.ProductWithExternalDemandIndex[p]
        result = self.StartStock + w * self.NrStockVariablePerScenario + indexp
        return result

    # Return the name of the variable associated with the quanity of product p decided at the current stage
    def GetNameQuantityVariable2(self, p, t):
        return "Q_%d_%d"%(p, t)

    # Return the name of the variable associated with the stock of product p decided at the current stage
    def GetNameStockVariable2(self, p,  w):
        return "I_%d_%d"%(p,  w)

    # Return the name of the variable associated with the backorder of product p decided at the current stage
    def GetNameBackorderVariable2(self, p,  w):
         return "B_%d_%d" % (p,  w)

 #Define the variables
    def DefineVariables2(self):


        #Variable for the inventory
        self.Cplex.variables.add(obj=[math.pow(self.Instance.Gamma, t)
                                      * self.Instance.InventoryCosts[p]
                                      * self.FixedScenarioPobability[w]
                                      for p in self.Instance.ProductWithExternalDemand
                                      for t in self.GetLastStageTimeRangeStock(p)
                                      for w in self.FixedScenarioSet],
                                  lb=[0.0] * self.NrStockVariable,
                                  ub=[self.M] * self.NrStockVariable)

        # Backorder/lostsales variables

        backordercost = [math.pow(self.Instance.Gamma, t) * self.Instance.LostSaleCost[p]
                         * self.FixedScenarioPobability[w]
                         for p in self.Instance.ProductWithExternalDemand
                         for t in self.GetLastStageTimeRangeStock(p)
                         for w in self.FixedScenarioSet]

        self.Cplex.variables.add(obj=backordercost,
                                 lb=[0.0] * self.NrBackOrderVariable,
                                 ub=[self.M] * self.NrBackOrderVariable)

        # Compute the Flow from previous stage
        flowfromprevioustage = [self.GetFlowFromPreviousStage(p) for p in self.Instance.ProductWithExternalDemand]
        self.Cplex.variables.add(obj=[0.0] * self.NRFlowFromPreviousStage,
                                 lb=flowfromprevioustage,
                                 ub=flowfromprevioustage)
        #In debug mode, the variables have names
        if Constants.Debug:
            self.AddVariableName()

    def IsLastStage(self):
        return True

    def IsFirstStage(self):
        return False

    def IsPenultimateStage(self):
        return False

    def AddVariableName2(self):
        if Constants.Debug:
            print("Add the names of the variable")
        # Define the variable name.
        # Usefull for debuging purpose. Otherwise, disable it, it is time consuming.
        if Constants.Debug:
            inventoryvars = []
            backordervars = []

            for w in self.FixedScenarioSet:
                for p in self.Instance.ProductWithExternalDemand:
                    for t in self.GetLastStageTimeRangeStock(p):
                        backordervars.append((self.GetIndexBackorderVariable(p, w), self.GetNameBackorderVariable(p,  w)))

                for p in self.Instance.ProductSet:
                    for t in self.GetLastStageTimeRangeStock(p):
                        inventoryvars.append((self.GetIndexStockVariable(p,  w), self.GetNameStockVariable(p,  w)))

            inventoryvars = list(set(inventoryvars))
            backordervars = list(set(backordervars))
            varnames = inventoryvars + backordervars
            if Constants.Debug:
                print("Variable Names: %s"%varnames)
            self.Cplex.variables.set_names(varnames)


    def CreateProductionConstraints(self):
        print("Not in last period")

    def CreateCapacityConstraints(self):
        # Capacity constraint
        if self.Instance.NrResource > 0:
            for w in self.FixedScenarioSet:
                for k in range(self.Instance.NrResource):
                    vars = [self.GetIndexQuantityVariable(p,w) for p in self.Instance.ProductSet]
                    coeff = [self.Instance.ProcessingTime[p][k] for p in self.Instance.ProductSet]
                    righthandside = [self.Instance.Capacity[k]]
                    self.Cplex.linear_constraints.add(lin_expr=[cplex.SparsePair(vars, coeff)],
                                                      senses=["L"],
                                                      rhs=righthandside)
                    if Constants.Debug:
                        self.Stage.Cplex.linear_constraints.set_names(self.LastAddedConstraintIndex,
                                                                      "Capt%d" % ( k))

                    self.IndexCapacityConstraint.append(self.LastAddedConstraintIndex)
                    self.IndexCapacityConstraintPerScenario[w].append(self.LastAddedConstraintIndex)
                    self.LastAddedConstraintIndex = self.LastAddedConstraintIndex + 1

                    self.ConcernedResourceCapacityConstraint.append(k)
                    self.ConcernedScenarioCapacityConstraint.append(w)


    def GetVariableValue2(self, sol):


        #print("scenario is zero because this is only implemented for forward pass")
        indexarray = [self.GetIndexStockVariable(p,  0) for p in self.Instance.ProductSet for t in self.GetLastStageTimeRangeStock(p)]
        inventory = sol.get_values(indexarray)

        indexarray = [self.GetIndexBackorderVariable(p,  0) for p in self.Instance.ProductSet for t in self.GetLastStageTimeRangeStock(p)]
        backorder = sol.get_values(indexarray)

        self.InventoryValue[self.CurrentTrialNr] = ['nan' for p in self.Instance.ProductSet]
        self.BackorderValue[self.CurrentTrialNr] = ['nan' for p in self.Instance.ProductWithExternalDemand]
        for p in self.Instance.ProductSet:
            indexp = self.Instance.ProductWithExternalDemandIndex[p]

            for t in self.GetLastStageTimeRangeStock(p):
                self.InventoryValue[self.CurrentTrialNr][p] = inventory[indexp]

                if self.Instance.HasExternalDemand[p]:
                    self.BackorderValue[self.CurrentTrialNr][indexp] = backorder[indexp]


    # Demand and materials requirement: set the value of the invetory level and backorder quantity according to
    #  the quantities produced and the demand
    def CreateFlowConstraints2(self):
        self.FlowConstraintNR = [["" for t in self.Instance.TimeBucketSet] for p in self.Instance.ProductSet]
        for w in self.FixedScenarioSet:
            for p in self.Instance.ProductSet:
                for t in self.GetLastStageTimeRangeStock(p):
                    if self.Instance.HasExternalDemand[p]:
                    # To speed up the creation of the model, only the variable and coffectiant which were not in the previous constraints are added (See the model definition)
                    #for t in self.GetLastStageTimeRangeStock(p):
                        righthandside = [self.GetRHSFlow(p, w, self.IsForward)]
                        backordervar = []
                        backordercoeff =[]
                        if self.Instance.HasExternalDemand[p]:
                            backordervar = [self.GetIndexBackorderVariable(p,  w)]
                            backordercoeff = [1]


                        inventoryvar = [self.GetIndexStockVariable(p,  w)]
                        inventorycoeff = [-1]

                        flowfrompreviousstagevar = [self.GetIndexFlowFromPreviousStage(p)]
                        flowfrompreviousstagecoeff = [-1]

                        vars = inventoryvar + backordervar + flowfrompreviousstagevar #+ quantityvar + dependentdemandvar
                        coeff = inventorycoeff + backordercoeff + flowfrompreviousstagecoeff #+ quantityvarceoff + dependentdemandvarcoeff

                        if len(vars) > 0:
                                self.Cplex.linear_constraints.add(lin_expr=[cplex.SparsePair(vars, coeff)],
                                                                  senses=["E"],
                                                                  rhs=righthandside)
                        self.FlowConstraintNR[p][t] = "Flowp%dy%d"%(p, t)
                        if Constants.Debug:
                            self.Cplex.linear_constraints.set_names(self.LastAddedConstraintIndex,
                                                                          self.FlowConstraintNR[p][t])

                        self.IndexFlowConstraint.append(self.LastAddedConstraintIndex)
                        self.IndexFlowConstraintPerScenario[w].append(self.LastAddedConstraintIndex)
                        self.LastAddedConstraintIndex = self.LastAddedConstraintIndex + 1

                        self.ConcernedTimeFlowConstraint.append(self.GetTimePeriodAssociatedToInventoryVariable(p,t))
                        self.ConcernedProductFlowConstraint.append(p)
                        self.ConcernedScenarioFlowConstraint.append(w)


    def IncreaseCutWithFlowDual2(self, cut, sol):
        if Constants.Debug:
            print("Increase cut with flow dual")
        duals = sol.get_dual_values(self.IndexFlowConstraint)
        for i in range(len(duals)):
            scenario = self.ConcernedScenarioFlowConstraint[i]
            duals[i] = duals[i]
            p = self.ConcernedProductFlowConstraint[i]
            periodproduction = self.ConcernedTimeFlowConstraint[i] - self.Instance.LeadTimes[p]
            if periodproduction >= 0:
                cut.IncreaseCoefficientQuantity(p, periodproduction, duals[i])

            periodpreviousstock = self.ConcernedTimeFlowConstraint[i] - 1

            if periodpreviousstock < self.GetStartStageTimeRangeStock(p):
                if periodpreviousstock >= 0:
                    cut.IncreaseCoefficientInventory(p, self.GetTimePeriodAssociatedToInventoryVariable(p) - 1,
                                                             duals[i])
                else:
                    cut.IncreaseInitInventryRHS(-1 * duals[i] * self.Instance.StartingInventories[p])

            if self.Instance.HasExternalDemand[p]:
                cut.IncreaseCoefficientBackorder(p, periodpreviousstock,  -duals[i])
                cut.IncreaseDemandRHS(duals[i] * self.SDDPOwner.SetOfSAAScenario[scenario].Demands[self.ConcernedTimeFlowConstraint[i]][p])

