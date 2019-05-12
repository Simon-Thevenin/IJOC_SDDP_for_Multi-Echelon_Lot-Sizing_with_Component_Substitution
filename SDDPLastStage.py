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

    #def GetIndexBackorderVariable(self, p, t, w):
    #    indexp = self.Instance.ProductWithExternalDemandIndex[p]
    #    return self.StartBackOrder + w*self.NrBackOrderVariablePerScenario + indexp * self.GetNrStageStock(p) + self.rescaletimestock(t, p)

    def GetIndexBackorderVariable2(self, p,  w):
        indexp = self.Instance.ProductWithExternalDemandIndex[p]
        return self.StartBackOrder + w * self.NrBackOrderVariablePerScenario + indexp

    # Return the index of the variable associated with the quanity of product p decided at the current stage
    def GetIndexQuantityVariable2(self, p, w):
        return self.StartQuantity + w*self.NrQuantityVariable + p * self.Instance.NrTimeBucketWithoutUncertaintyAfter + p #self.rescaletimequantity(t)

    # Return the index of the variable associated with the stock of product p decided at the current stage
    #def GetIndexStockVariable(self, p, t, w):
    #    result = sum(self.GetNrStageStock(q) for q in range(0, p))
    #    result = self.StartStock + w* self.NrStockVariablePerScenario+ result + self.rescaletimestock(t, p)
    #    return result

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

        #Variable for the production quanitity
        #self.Cplex.variables.add(obj=[0.0] * self.NrQuantityVariable,
        #                         lb=[0.0] * self.NrQuantityVariable,
        #                         ub=[self.M] * self.NrQuantityVariable)

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
        #backordertime = self.GetLastStageTimeRangeStock(p)
        #backordertime.pop()

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
            #quantityvars = []
            inventoryvars = []
            backordervars = []

            #for t in self.GetLastStageTimeRangeQuantity():
            #    for p in self.Instance.ProductSet:
            #        quantityvars.append((self.GetIndexQuantityVariable(p, t), self.GetNameQuantityVariable(p, t)))
            for w in self.FixedScenarioSet:
                for p in self.Instance.ProductWithExternalDemand:
                    for t in self.GetLastStageTimeRangeStock(p):
                        backordervars.append((self.GetIndexBackorderVariable(p, w), self.GetNameBackorderVariable(p,  w)))

                for p in self.Instance.ProductSet:
                    for t in self.GetLastStageTimeRangeStock(p):
                        inventoryvars.append((self.GetIndexStockVariable(p,  w), self.GetNameStockVariable(p,  w)))

            #quantityvars = list(set(quantityvars))
            inventoryvars = list(set(inventoryvars))
            backordervars = list(set(backordervars))
            varnames = inventoryvars + backordervars
            if Constants.Debug:
                print("Variable Names: %s"%varnames)
            self.Cplex.variables.set_names(varnames)


    #This function returns the right hand side of the production consraint associated with product p
    #def GetProductionConstrainRHS(self, p, t):
    #    yvalue = self.SDDPOwner.GetSetupFixedEarlier(p, t, self.CurrentTrialNr)
    #    righthandside = self.GetBigMValue(p) * yvalue
    #    return righthandside

    def CreateProductionConstraints(self):
        print("Not in last period")
        #for p in self.Instance.ProductSet:
        #    for t in self.GetLastStageTimeRangeQuantity():
        #        righthandside = [self.GetProductionConstrainRHS(p,t)]

        #        vars = [self.GetIndexQuantityVariable(p, t)]
        #        coeff = [1.0]

        #        self.Cplex.linear_constraints.add(lin_expr=[cplex.SparsePair(vars, coeff)],
        #                                              senses=["L"],
        #                                              rhs=righthandside,
        #                                              names= ["Prodp%dt%d"%(p,t)])

        #        self.IndexProductionQuantityConstraint.append(self.LastAddedConstraintIndex)
        #        self.LastAddedConstraintIndex = self.LastAddedConstraintIndex + 1
        #        self.ConcernedProductProductionQuantityConstraint.append(p)
        #        self.ConcernedTimeProductionQuantityConstraint.append(t)

    def CreateCapacityConstraints(self):
        # Capacity constraint
        if self.Instance.NrResource > 0:
            for w in self.FixedScenarioSet:
            #for t in self.GetLastStageTimeRangeQuantity():
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

        #indexarray = [self.GetIndexQuantityVariable(p, t) for t in self.GetLastStageTimeRangeQuantity() for p in self.Instance.ProductSet]
        #self.QuantityValues[self.CurrentScenarioNr] = sol.get_values(indexarray)
        #self.QuantityValues[self.CurrentScenarioNr] = np.array(self.QuantityValues[self.CurrentScenarioNr], np.float64).reshape(
        #                                                         (self.GetNrStageQuantity(), self.Instance.NrProduct, )).tolist()

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
                #indexi = sum(self.GetNrStageStock(q) for q in range(p)) + self.rescaletimestock(t,p)
                self.InventoryValue[self.CurrentTrialNr][p] = inventory[indexp]

                if self.Instance.HasExternalDemand[p]:
                    #indexb = sum(self.GetNrStageStock(q) for q in range(p) if self.Instance.HasExternalDemand[p]) + self.rescaletimestock(t, p)
                    self.BackorderValue[self.CurrentTrialNr][indexp] = backorder[indexp]

    #def IncreaseCutWithFlowDual(self, cut, solution):
    #    print "Increase the cut with flow constraint of the last stage"


   # def GetRHSFlowConst(self, p, t):
   #     righthandside = 0
   #     # Get the level of inventory computed in the previsous stage
   #     if t == self.GetStartStageTimeRangeStock(p): #if this t is the first time period with inventory variable
   #         previousperiod = t - 1
   #         if self.Instance.HasExternalDemand[p]:
   #             righthandside = righthandside - 1 * self.SDDPOwner.GetInventoryFixedEarlier(p, previousperiod, self.CurrentScenarioNr)
   #             righthandside = righthandside + self.SDDPOwner.GetBackorderFixedEarlier(p, previousperiod, self.CurrentScenarioNr)
   #         else:
   #             righthandside = -1 * self.SDDPOwner.GetInventoryFixedEarlier(p,previousperiod, self.CurrentScenarioNr)

      #  for t2 in range(self.GetStartStageTimeRangeStock(p), t ):
    #    righthandside= righthandside + self.SDDPOwner.CurrentSetOfScenarios[self.CurrentScenarioNr].Demands[t][p]

    #    productionstartedtime = t - self.Instance.LeadTimes[p]
    #    if productionstartedtime < self.GetStartStageTimeRangeQuantity():
    #        righthandside = righthandside \
    #                        - self.SDDPOwner.GetQuantityFixedEarlier(p, productionstartedtime,
      #                                                               self.CurrentScenarioNr)
    #    return righthandside


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
                        #quantityvar = []
                        #quantityvarceoff = []
                        #dependentdemandvar = []
                        #dependentdemandvarcoeff = []
                        if self.Instance.HasExternalDemand[p]:
                            backordervar = [self.GetIndexBackorderVariable(p,  w)]
                            backordercoeff = [1]

                        #if t - self.Instance.LeadTimes[p] >= self.GetStartStageTimeRangeQuantity():
                        #        quantityvar = quantityvar + [self.GetIndexQuantityVariable(p, t - self.Instance.LeadTimes[p])]
                        #        quantityvarceoff = quantityvarceoff + [1]

                        #dependentdemandvar = dependentdemandvar + [self.GetIndexQuantityVariable(q, t) for q in
                        #                                            self.Instance.RequieredProduct[p]]

                        #dependentdemandvarcoeff = dependentdemandvarcoeff + [-1 * self.Instance.Requirements[q][p] for q in
                        #                                                     self.Instance.RequieredProduct[p] ]

                        inventoryvar = [self.GetIndexStockVariable(p,  w)]
                        inventorycoeff = [-1]

                        flowfrompreviousstagevar = [self.GetIndexFlowFromPreviousStage(p)]
                        flowfrompreviousstagecoeff = [-1]
                    #if t > self.GetStartStageTimeRangeStock(p):
                        #    inventoryvar = inventoryvar + [self.GetIndexStockVariable(p, t-1,w)]
                        #    inventorycoeff = inventorycoeff +[1]
                        #    if self.Instance.HasExternalDemand[p]:
                        #        backordervar = backordervar + [self.GetIndexBackorderVariable(p, t-1,w)]
                        #        backordercoeff = backordercoeff +[-1]
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
            duals[i] = duals[i] #*self.SDDPOwner.SetOfSAAScenario[scenario].Probability

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

